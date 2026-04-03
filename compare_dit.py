"""
Comparison script: Original DiT (DDPM+SVD) vs ResShift DiT.

Trains both models on the same dataset, evaluates FID periodically,
and generates side-by-side comparison images & metrics.

Usage:
    python compare_dit.py \
        --data-path ./imagenet100/train \
        --num-classes 100 \
        --epochs 5 \
        --fid-every 2000 \
        --fid-samples 1024

This script runs sequentially (not DDP) for simplicity.
For multi-GPU training, use train.py and train_resshift.py separately.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
from copy import deepcopy
from time import time
from PIL import Image
import numpy as np
import argparse
import json
import os

from models import DiT_models
from diffusion import create_diffusion, create_resshift_diffusion
from diffusers.models import AutoencoderKL
from fid_utils import (
    compute_fid_score, generate_samples_ddpm, generate_samples_resshift,
    save_samples_to_dir, svd_lowrank,
)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def train_one_step_ddpm(model, ema, vae, diffusion, x, y, opt, device):
    """One training step for the original DiT (DDPM with SVD x_cond)."""
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.sample().mul_(0.18215)
        half_T = int(0.5 * diffusion.num_timesteps)
        t_half = torch.full((z.shape[0],), half_T, dtype=torch.long, device=device)
        z_noisy = diffusion.q_sample(z, t_half)
        x_cond = svd_lowrank(z_noisy, ratio=0.25)

    t = torch.randint(0, diffusion.num_timesteps, (z.shape[0],), device=device)
    model_kwargs = dict(y=y, x_cond=x_cond)
    loss_dict = diffusion.training_losses(model, z, t, model_kwargs)
    loss = loss_dict["loss"].mean()

    opt.zero_grad()
    loss.backward()
    opt.step()
    update_ema(ema, model)
    return loss.item()


def train_one_step_resshift(model, ema, vae, ddpm_diffusion, resshift_diffusion,
                            x, y, opt, device, svd_ratio=0.25):
    """One training step for ResShift DiT."""
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        z_clean = vae.encode(x).latent_dist.sample().mul_(0.18215)
        half_T = int(0.5 * ddpm_diffusion.num_timesteps)
        t_half = torch.full((z_clean.shape[0],), half_T, dtype=torch.long, device=device)
        z_noisy = ddpm_diffusion.q_sample(z_clean, t_half)
        x_cond = svd_lowrank(z_noisy, ratio=svd_ratio)

    t = torch.randint(0, resshift_diffusion.num_timesteps, (z_clean.shape[0],), device=device)
    model_kwargs = dict(y=y, x_cond=x_cond)
    loss_dict = resshift_diffusion.training_losses(
        model, x_start=z_clean, y=x_cond, t=t, model_kwargs=model_kwargs,
    )
    loss = loss_dict["loss"].mean()

    opt.zero_grad()
    loss.backward()
    opt.step()
    update_ema(ema, model)
    return loss.item()


def generate_comparison_images(ema_ddpm, ema_resshift, vae, num_classes,
                               latent_size, device, class_labels, cfg_scale=4.0,
                               ddpm_steps=250, n_timestep=15, kappa=1.0,
                               svd_ratio=0.25):
    """Generate images from both models for visual comparison."""
    n = len(class_labels)
    y = torch.tensor(class_labels, device=device)

    # --- Original DiT (DDPM) sampling ---
    diffusion = create_diffusion(str(ddpm_steps))
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    z_cfg = torch.cat([z, z], 0)
    y_null = torch.tensor([num_classes] * n, device=device)
    y_cfg = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y_cfg, cfg_scale=cfg_scale)

    ema_ddpm.eval()
    samples_ddpm = diffusion.p_sample_loop(
        ema_ddpm.forward_with_cfg, z_cfg.shape, z_cfg,
        clip_denoised=False, model_kwargs=model_kwargs,
        progress=True, device=device,
    )
    samples_ddpm = samples_ddpm[:n]
    samples_ddpm = vae.decode(samples_ddpm / 0.18215).sample

    # --- ResShift DiT sampling ---
    resshift_diffusion = create_resshift_diffusion(n_timestep=n_timestep, kappa=kappa)

    # Stage 1: DDPM to 0.5T (reuse same noise for fair comparison)
    z2 = torch.randn(n, 4, latent_size, latent_size, device=device)
    z2_cfg = torch.cat([z2, z2], 0)
    model_kwargs2 = dict(y=y_cfg, cfg_scale=cfg_scale)

    ema_resshift.eval()
    z_half = diffusion.p_sample_loop(
        ema_resshift.forward_with_cfg, z2_cfg.shape, z2_cfg,
        clip_denoised=False, model_kwargs=model_kwargs2,
        progress=True, device=device,
        clip_point=0.5,
    )
    z_half = z_half[:n]

    # Stage 2: SVD
    x_cond = svd_lowrank(z_half, ratio=svd_ratio)

    # Stage 3: ResShift
    model_kwargs_rs = dict(y=y, x_cond=x_cond)
    z_clean = resshift_diffusion.p_sample_loop(
        ema_resshift, x_cond,
        clip_denoised=False, model_kwargs=model_kwargs_rs,
        device=device, progress=True,
    )
    samples_resshift = vae.decode(z_clean / 0.18215).sample

    return samples_ddpm, samples_resshift


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    latent_size = args.image_size // 8

    # --- Create two identical models ---
    model_ddpm = DiT_models[args.model](
        input_size=latent_size, num_classes=args.num_classes
    ).to(device)
    model_resshift = DiT_models[args.model](
        input_size=latent_size, num_classes=args.num_classes
    ).to(device)

    # Sync initial weights
    model_resshift.load_state_dict(model_ddpm.state_dict())

    ema_ddpm = deepcopy(model_ddpm)
    ema_resshift = deepcopy(model_resshift)
    ema_ddpm.eval()
    ema_resshift.eval()

    # Diffusion processes
    ddpm_diffusion = create_diffusion(timestep_respacing="")
    resshift_diffusion = create_resshift_diffusion(
        n_timestep=args.n_timestep, kappa=args.kappa
    )

    # VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Optimizers
    opt_ddpm = torch.optim.AdamW(model_ddpm.parameters(), lr=1e-4, weight_decay=0)
    opt_resshift = torch.optim.AdamW(model_resshift.parameters(), lr=1e-4, weight_decay=0)

    # Dataset
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3, inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    print(f"Dataset: {len(dataset)} images, {args.num_classes} classes")
    print(f"Model: {args.model}, params: {sum(p.numel() for p in model_ddpm.parameters()):,}")

    # Metrics tracking
    metrics = {
        "steps": [], "ddpm_loss": [], "resshift_loss": [],
        "ddpm_fid": [], "resshift_fid": [], "fid_steps": [],
    }
    best_fid_ddpm = float('inf')
    best_fid_resshift = float('inf')

    train_steps = 0
    model_ddpm.train()
    model_resshift.train()

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        running_ddpm = 0
        running_resshift = 0
        n_batches = 0

        for x, y_label in loader:
            # Train original DiT (DDPM with SVD x_cond)
            loss_d = train_one_step_ddpm(
                model_ddpm, ema_ddpm, vae, ddpm_diffusion,
                x, y_label, opt_ddpm, device,
            )

            # Train ResShift DiT
            loss_r = train_one_step_resshift(
                model_resshift, ema_resshift, vae, ddpm_diffusion,
                resshift_diffusion, x, y_label, opt_resshift, device,
                svd_ratio=args.svd_ratio,
            )

            running_ddpm += loss_d
            running_resshift += loss_r
            n_batches += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                avg_d = running_ddpm / n_batches
                avg_r = running_resshift / n_batches
                print(f"  step={train_steps:06d} | DDPM loss: {avg_d:.4f} | ResShift loss: {avg_r:.4f}")
                metrics["steps"].append(train_steps)
                metrics["ddpm_loss"].append(avg_d)
                metrics["resshift_loss"].append(avg_r)
                running_ddpm = 0
                running_resshift = 0
                n_batches = 0

            # FID evaluation
            if args.fid_every > 0 and train_steps % args.fid_every == 0 and train_steps > 0:
                print(f"\n  Computing FID at step {train_steps}...")

                fid_d = compute_fid_score(
                    ema_ddpm, vae, args.data_path, device,
                    num_samples=args.fid_samples, mode="ddpm",
                    cfg_scale=4.0, num_classes=args.num_classes,
                    latent_size=latent_size, batch_size=16,
                    tmp_dir=f"{args.output_dir}/fid_tmp_ddpm",
                    ddpm_steps=250,
                )
                fid_r = compute_fid_score(
                    ema_resshift, vae, args.data_path, device,
                    num_samples=args.fid_samples, mode="resshift",
                    cfg_scale=4.0, num_classes=args.num_classes,
                    latent_size=latent_size, batch_size=16,
                    tmp_dir=f"{args.output_dir}/fid_tmp_resshift",
                    ddpm_steps=250, n_timestep=args.n_timestep,
                    kappa=args.kappa, svd_ratio=args.svd_ratio,
                )
                print(f"  FID: DDPM={fid_d:.2f} (best={best_fid_ddpm:.2f}) | "
                      f"ResShift={fid_r:.2f} (best={best_fid_resshift:.2f})")

                metrics["fid_steps"].append(train_steps)
                metrics["ddpm_fid"].append(fid_d)
                metrics["resshift_fid"].append(fid_r)

                # Save best checkpoints
                if fid_d < best_fid_ddpm:
                    best_fid_ddpm = fid_d
                    torch.save({
                        "ema": ema_ddpm.state_dict(),
                        "fid": fid_d, "step": train_steps,
                    }, f"{args.output_dir}/best_ddpm.pt")
                    print(f"  New best DDPM FID: {fid_d:.2f}")

                if fid_r < best_fid_resshift:
                    best_fid_resshift = fid_r
                    torch.save({
                        "ema": ema_resshift.state_dict(),
                        "fid": fid_r, "step": train_steps,
                    }, f"{args.output_dir}/best_resshift.pt")
                    print(f"  New best ResShift FID: {fid_r:.2f}")

                # Save metrics snapshot
                with open(f"{args.output_dir}/metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)

                model_ddpm.train()
                model_resshift.train()

    # --- Final comparison ---
    print("\n=== Generating comparison images ===")
    class_labels = list(range(min(8, args.num_classes)))
    with torch.no_grad():
        samples_ddpm, samples_resshift = generate_comparison_images(
            ema_ddpm, ema_resshift, vae, args.num_classes,
            latent_size, device, class_labels,
            cfg_scale=4.0, ddpm_steps=250,
            n_timestep=args.n_timestep, kappa=args.kappa,
            svd_ratio=args.svd_ratio,
        )

    save_image(samples_ddpm, f"{args.output_dir}/comparison_ddpm.png",
               nrow=4, normalize=True, value_range=(-1, 1))
    save_image(samples_resshift, f"{args.output_dir}/comparison_resshift.png",
               nrow=4, normalize=True, value_range=(-1, 1))

    # Save final metrics
    with open(f"{args.output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Best FID - DDPM:     {best_fid_ddpm:.2f}")
    print(f"Best FID - ResShift: {best_fid_resshift:.2f}")
    print(f"Results saved to: {args.output_dir}/")
    print(f"  comparison_ddpm.png     - Original DiT samples")
    print(f"  comparison_resshift.png - ResShift DiT samples")
    print(f"  metrics.json            - Training & FID metrics")
    print(f"  best_ddpm.pt            - Best DDPM checkpoint")
    print(f"  best_resshift.pt        - Best ResShift checkpoint")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="comparison_results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--fid-every", type=int, default=2000,
                        help="Compute FID every N steps (0=disabled)")
    parser.add_argument("--fid-samples", type=int, default=1024,
                        help="Number of samples for FID")
    parser.add_argument("--n-timestep", type=int, default=15)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--svd-ratio", type=float, default=0.25)
    args = parser.parse_args()
    main(args)
