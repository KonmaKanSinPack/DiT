"""
Comparison script: Original DiT (DDPM+SVD) vs ResShift DiT (DDPM→SVD→ResShift).

This script:
  1. Trains both models on the same data for the same number of steps
  2. Samples from both using the same random seed and class labels
  3. Produces side-by-side visual comparison and quantitative metrics

Usage:
  # Full pipeline: train both + sample + compare
  python compare_dit_resshift.py --data-path ./imagenet100/train --num-classes 100 --train-steps 500

  # Compare from existing checkpoints (skip training)
  python compare_dit_resshift.py --skip-training \
      --orig-ckpt results/000-DiT-XL-2/checkpoints/0000500.pt \
      --resshift-ckpt results_resshift/000-DiT-XL-2-resshift/checkpoints/0000200.pt \
      --num-classes 100

  # Only train (skip sampling/comparison)
  python compare_dit_resshift.py --data-path ./imagenet100/train --num-classes 100 --only-train --train-steps 500
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from time import time
from glob import glob
import argparse
import logging
import os
import json
import sys

from models import DiT_models
from diffusion import create_diffusion, create_resshift_diffusion
from diffusers.models import AutoencoderKL
from download import find_model


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


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


def svd_lowrank(x, ratio=0.25):
    """SVD low-rank reconstruction keeping top `ratio` fraction of singular values."""
    U, S, Vt = torch.linalg.svd(x)
    r_use = max(int(ratio * S.size(-1)), 1)
    Sr = S[:, :, :r_use]
    recon = (U[:, :, :, :r_use] * Sr.unsqueeze(-2)) @ Vt[:, :, :r_use, :]
    return recon


def setup_logger(log_dir, name="compare"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(f"{log_dir}/compare_log.txt"))
    fmt = logging.Formatter('[\033[34m%(asctime)s\033[0m] %(message)s', '%Y-%m-%d %H:%M:%S')
    for h in logger.handlers:
        h.setFormatter(fmt)
    return logger


# ---------------------------------------------------------------------------
# Phase 1: Training (both models, same data, same steps)
# ---------------------------------------------------------------------------

def train_original_dit(args, logger, device, rank):
    """Train original DiT with DDPM + SVD x_cond (matching train.py logic)."""
    logger.info("=" * 60)
    logger.info("Training ORIGINAL DiT (DDPM + SVD x_cond)")
    logger.info("=" * 60)

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    )

    if args.pretrained_ckpt:
        state_dict = find_model(args.pretrained_ckpt)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded pretrained weights: {args.pretrained_ckpt}")

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # 1000 steps
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank,
                                 shuffle=True, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    logger.info(f"Dataset: {len(dataset):,} images")

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # Checkpoint dir
    ckpt_dir = os.path.join(args.output_dir, "orig_dit", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_steps = 0
    running_loss = 0
    log_steps = 0
    start_time = time()
    losses = []

    for epoch in range(args.max_epochs):
        sampler.set_epoch(epoch)
        for x, y in loader:
            if train_steps >= args.train_steps:
                break

            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                z_clean = vae.encode(x).latent_dist.sample().mul_(0.18215)

                # SVD degradation (same as train.py): q_sample at 0.5T + SVD 25%
                half_T = int(0.5 * diffusion.num_timesteps)
                t_half = torch.full((z_clean.shape[0],), half_T, dtype=torch.long, device=device)
                z_noisy = diffusion.q_sample(z_clean, t_half)
                x_cond = svd_lowrank(z_noisy, ratio=args.svd_ratio)

            t = torch.randint(0, diffusion.num_timesteps, (z_clean.shape[0],), device=device)
            model_kwargs = dict(y=y, x_cond=x_cond)
            loss_dict = diffusion.training_losses(model, z_clean, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            losses.append(loss.item())

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                elapsed = time() - start_time
                avg_loss = running_loss / log_steps
                logger.info(f"[OrigDiT] step={train_steps:05d} loss={avg_loss:.4f} "
                            f"speed={log_steps / elapsed:.1f} steps/s")
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and rank == 0:
                ckpt = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                    "step": train_steps,
                }
                path = f"{ckpt_dir}/{train_steps:07d}.pt"
                torch.save(ckpt, path)
                logger.info(f"[OrigDiT] Saved checkpoint: {path}")

        if train_steps >= args.train_steps:
            break

    # Final save
    if rank == 0:
        final_path = f"{ckpt_dir}/final.pt"
        torch.save({
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args,
            "step": train_steps,
        }, final_path)
        logger.info(f"[OrigDiT] Final checkpoint: {final_path} ({train_steps} steps)")

    del model, ema, vae, opt
    torch.cuda.empty_cache()
    return final_path, losses


def train_resshift_dit(args, logger, device, rank):
    """Train ResShift DiT with DDPM→SVD→ResShift pipeline."""
    logger.info("=" * 60)
    logger.info("Training RESSHIFT DiT (DDPM → SVD → ResShift)")
    logger.info("=" * 60)

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    )

    if args.pretrained_ckpt:
        state_dict = find_model(args.pretrained_ckpt)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded pretrained weights: {args.pretrained_ckpt}")

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])

    ddpm_diffusion = create_diffusion(timestep_respacing="")  # 1000 steps
    resshift_diffusion = create_resshift_diffusion(
        n_timestep=args.n_timestep, kappa=args.kappa,
    )
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"ResShift: n_timestep={args.n_timestep}, kappa={args.kappa}, svd_ratio={args.svd_ratio}")

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank,
                                 shuffle=True, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    logger.info(f"Dataset: {len(dataset):,} images")

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    ckpt_dir = os.path.join(args.output_dir, "resshift_dit", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_steps = 0
    running_loss = 0
    log_steps = 0
    start_time = time()
    losses = []

    for epoch in range(args.max_epochs):
        sampler.set_epoch(epoch)
        for x, y in loader:
            if train_steps >= args.train_steps:
                break

            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                z_clean = vae.encode(x).latent_dist.sample().mul_(0.18215)

                # DDPM forward at 0.5T + SVD → x_cond
                half_T = int(0.5 * ddpm_diffusion.num_timesteps)
                t_half = torch.full((z_clean.shape[0],), half_T, dtype=torch.long, device=device)
                z_noisy = ddpm_diffusion.q_sample(z_clean, t_half)
                x_cond = svd_lowrank(z_noisy, ratio=args.svd_ratio)

            t = torch.randint(0, resshift_diffusion.num_timesteps, (z_clean.shape[0],), device=device)
            model_kwargs = dict(y=y, x_cond=x_cond)
            loss_dict = resshift_diffusion.training_losses(
                model, x_start=z_clean, y=x_cond, t=t, model_kwargs=model_kwargs,
            )
            loss = loss_dict["loss"].mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            losses.append(loss.item())

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                elapsed = time() - start_time
                avg_loss = running_loss / log_steps
                logger.info(f"[ResShift] step={train_steps:05d} loss={avg_loss:.4f} "
                            f"speed={log_steps / elapsed:.1f} steps/s")
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and rank == 0:
                ckpt = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                    "step": train_steps,
                }
                path = f"{ckpt_dir}/{train_steps:07d}.pt"
                torch.save(ckpt, path)
                logger.info(f"[ResShift] Saved checkpoint: {path}")

        if train_steps >= args.train_steps:
            break

    if rank == 0:
        final_path = f"{ckpt_dir}/final.pt"
        torch.save({
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args,
            "step": train_steps,
        }, final_path)
        logger.info(f"[ResShift] Final checkpoint: {final_path} ({train_steps} steps)")

    del model, ema, vae, opt
    torch.cuda.empty_cache()
    return final_path, losses


# ---------------------------------------------------------------------------
# Phase 2: Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_original_dit(ckpt_path, args, logger, device):
    """Sample from original DiT using standard DDPM reverse + SVD post-processing."""
    logger.info("=" * 60)
    logger.info("Sampling from ORIGINAL DiT")
    logger.info("=" * 60)

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.sample_vae}").to(device)

    class_labels = args.class_labels
    n = len(class_labels)
    y = torch.tensor(class_labels, device=device)

    # Standard DDPM sampling with CFG
    z = torch.randn(n, 4, latent_size, latent_size, device=device,
                     generator=torch.Generator(device).manual_seed(args.seed))
    z_cfg = torch.cat([z, z], 0)
    y_null = torch.tensor([args.num_classes] * n, device=device)
    y_cfg = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y_cfg, cfg_scale=args.cfg_scale)

    logger.info(f"Running DDPM sampling ({args.sampling_steps} steps, cfg={args.cfg_scale})...")
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z_cfg.shape, z_cfg,
        clip_denoised=False, model_kwargs=model_kwargs,
        progress=True, device=device,
    )

    # SVD post-processing (matching sample.py logic)
    U, S, Vt = torch.linalg.svd(samples)
    r_use = max(int(args.svd_ratio * S.size(-1)), 1)
    Sr = S[:, :, :r_use]
    recon = (U[:, :, :, :r_use] * Sr.unsqueeze(-2)) @ Vt[:, :, :r_use, :]

    # Take conditional half only
    samples_cond, _ = samples.chunk(2, dim=0)
    recon_cond, _ = recon.chunk(2, dim=0)

    # VAE decode
    images_raw = vae.decode(samples_cond / 0.18215).sample
    images_svd = vae.decode(recon_cond / 0.18215).sample

    del model, vae
    torch.cuda.empty_cache()

    logger.info(f"[OrigDiT] Sampling done. Shape: {images_raw.shape}")
    return images_raw, images_svd


@torch.no_grad()
def sample_resshift_dit(ckpt_path, args, logger, device):
    """Sample from ResShift DiT: DDPM partial → SVD → ResShift → VAE decode."""
    logger.info("=" * 60)
    logger.info("Sampling from RESSHIFT DiT")
    logger.info("=" * 60)

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    ddpm_diffusion = create_diffusion(str(args.sampling_steps))
    resshift_diffusion = create_resshift_diffusion(
        n_timestep=args.n_timestep, kappa=args.kappa,
    )
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.sample_vae}").to(device)

    class_labels = args.class_labels
    n = len(class_labels)
    y = torch.tensor(class_labels, device=device)

    # Stage 1: DDPM partial reverse (T → 0.5T) with CFG
    z = torch.randn(n, 4, latent_size, latent_size, device=device,
                     generator=torch.Generator(device).manual_seed(args.seed))
    z_cfg = torch.cat([z, z], 0)
    y_null = torch.tensor([args.num_classes] * n, device=device)
    y_cfg = torch.cat([y, y_null], 0)
    model_kwargs_ddpm = dict(y=y_cfg, cfg_scale=args.cfg_scale)

    logger.info(f"Stage 1: DDPM partial sampling ({args.sampling_steps} steps, clip_point=0.5)...")
    z_half = ddpm_diffusion.p_sample_loop(
        model.forward_with_cfg, z_cfg.shape, z_cfg,
        clip_denoised=False, model_kwargs=model_kwargs_ddpm,
        progress=True, device=device,
        clip_point=0.5,
    )
    z_half = z_half[:n]  # take conditional half
    logger.info(f"Stage 1 done: z_half shape={z_half.shape}")

    # Stage 2: SVD low-rank
    x_cond = svd_lowrank(z_half, ratio=args.svd_ratio)
    logger.info(f"Stage 2 done: SVD x_cond shape={x_cond.shape}")

    # Stage 3: ResShift reverse
    model_kwargs_rs = dict(y=y, x_cond=x_cond)
    logger.info(f"Stage 3: ResShift sampling ({args.n_timestep} steps)...")
    z_clean = resshift_diffusion.p_sample_loop(
        model, x_cond,
        clip_denoised=False,
        model_kwargs=model_kwargs_rs,
        device=device, progress=True,
    )
    logger.info(f"Stage 3 done: z_clean shape={z_clean.shape}")

    # Stage 4: VAE decode
    images_final = vae.decode(z_clean / 0.18215).sample
    images_xcond = vae.decode(x_cond / 0.18215).sample

    del model, vae
    torch.cuda.empty_cache()

    logger.info(f"[ResShift] Sampling done. Shape: {images_final.shape}")
    return images_final, images_xcond


# ---------------------------------------------------------------------------
# Phase 3: Comparison & Visualization
# ---------------------------------------------------------------------------

def make_comparison_grid(orig_raw, orig_svd, rs_final, rs_xcond, save_dir, logger):
    """Create side-by-side visual comparison grids."""
    os.makedirs(save_dir, exist_ok=True)

    n = orig_raw.shape[0]
    nrow = min(n, 8)

    # Individual grids
    save_image(orig_raw, f"{save_dir}/orig_dit_raw.png", nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(orig_svd, f"{save_dir}/orig_dit_svd.png", nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(rs_final, f"{save_dir}/resshift_final.png", nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(rs_xcond, f"{save_dir}/resshift_xcond.png", nrow=nrow, normalize=True, value_range=(-1, 1))

    # Combined comparison: each row = [orig_raw, orig_svd, rs_xcond, rs_final]
    # Interleave samples for per-class comparison
    rows = []
    for i in range(n):
        rows.extend([orig_raw[i], orig_svd[i], rs_xcond[i], rs_final[i]])
    combined = torch.stack(rows)
    save_image(combined, f"{save_dir}/comparison_grid.png", nrow=4, normalize=True, value_range=(-1, 1),
               padding=2, pad_value=1.0)

    logger.info(f"Saved comparison images to {save_dir}/")
    logger.info(f"  comparison_grid.png: each row = [OrigDiT_raw, OrigDiT_SVD, ResShift_xcond, ResShift_final]")


def compute_metrics(orig_images, resshift_images, logger):
    """Compute basic quantitative metrics between the two methods."""
    # Pixel-space statistics
    orig_mean = orig_images.mean().item()
    orig_std = orig_images.std().item()
    rs_mean = resshift_images.mean().item()
    rs_std = resshift_images.std().item()

    # Per-sample L2 norms (measure of signal energy)
    orig_norms = orig_images.flatten(1).norm(dim=1)
    rs_norms = resshift_images.flatten(1).norm(dim=1)

    # Diversity: pairwise distance between samples
    def pairwise_dist(imgs):
        flat = imgs.flatten(1)
        n = flat.shape[0]
        if n < 2:
            return 0.0
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append((flat[i] - flat[j]).norm().item())
        return np.mean(dists)

    orig_diversity = pairwise_dist(orig_images)
    rs_diversity = pairwise_dist(resshift_images)

    metrics = {
        "orig_dit": {
            "pixel_mean": round(orig_mean, 4),
            "pixel_std": round(orig_std, 4),
            "avg_l2_norm": round(orig_norms.mean().item(), 4),
            "sample_diversity": round(orig_diversity, 4),
        },
        "resshift_dit": {
            "pixel_mean": round(rs_mean, 4),
            "pixel_std": round(rs_std, 4),
            "avg_l2_norm": round(rs_norms.mean().item(), 4),
            "sample_diversity": round(rs_diversity, 4),
        },
    }

    logger.info("=" * 60)
    logger.info("Quantitative Metrics")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<25} {'OrigDiT':>12} {'ResShift':>12}")
    logger.info("-" * 50)
    for key in metrics["orig_dit"]:
        v1 = metrics["orig_dit"][key]
        v2 = metrics["resshift_dit"][key]
        logger.info(f"{key:<25} {v1:>12.4f} {v2:>12.4f}")

    return metrics


def save_loss_curves(orig_losses, rs_losses, save_dir, logger):
    """Save loss curves as text (for plotting with external tools), and simple ASCII chart."""
    os.makedirs(save_dir, exist_ok=True)

    # Save raw data
    with open(f"{save_dir}/losses_orig.txt", "w") as f:
        for i, v in enumerate(orig_losses):
            f.write(f"{i + 1}\t{v:.6f}\n")
    with open(f"{save_dir}/losses_resshift.txt", "w") as f:
        for i, v in enumerate(rs_losses):
            f.write(f"{i + 1}\t{v:.6f}\n")

    # Summary stats
    def stats(losses, window=50):
        if len(losses) < window:
            window = len(losses)
        first = np.mean(losses[:window])
        last = np.mean(losses[-window:])
        return first, last, np.min(losses), np.mean(losses)

    o_first, o_last, o_min, o_avg = stats(orig_losses)
    r_first, r_last, r_min, r_avg = stats(rs_losses)

    logger.info("=" * 60)
    logger.info("Training Loss Summary")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<25} {'OrigDiT':>12} {'ResShift':>12}")
    logger.info("-" * 50)
    logger.info(f"{'First 50 steps avg':<25} {o_first:>12.4f} {r_first:>12.4f}")
    logger.info(f"{'Last 50 steps avg':<25} {o_last:>12.4f} {r_last:>12.4f}")
    logger.info(f"{'Min loss':<25} {o_min:>12.4f} {r_min:>12.4f}")
    logger.info(f"{'Overall avg':<25} {o_avg:>12.4f} {r_avg:>12.4f}")

    loss_summary = {
        "orig_dit": {"first_50_avg": o_first, "last_50_avg": o_last, "min": o_min, "avg": o_avg},
        "resshift_dit": {"first_50_avg": r_first, "last_50_avg": r_last, "min": r_min, "avg": r_avg},
    }

    with open(f"{save_dir}/loss_summary.json", "w") as f:
        json.dump(loss_summary, f, indent=2)

    return loss_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Setup DDP (needed even for single GPU)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.manual_seed(args.seed)
    torch.cuda.set_device(device)

    logger = setup_logger(args.output_dir)
    logger.info("=" * 60)
    logger.info("DiT vs ResShift DiT — Comparison Pipeline")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}, Image: {args.image_size}x{args.image_size}")
    logger.info(f"Classes: {args.num_classes}, Train steps: {args.train_steps}")
    logger.info(f"ResShift: n_timestep={args.n_timestep}, kappa={args.kappa}, svd={args.svd_ratio}")

    orig_losses, rs_losses = [], []
    orig_ckpt = args.orig_ckpt
    rs_ckpt = args.resshift_ckpt

    # ---- Phase 1: Training ----
    if not args.skip_training:
        logger.info("\n\n>>> PHASE 1: TRAINING <<<\n")
        orig_ckpt, orig_losses = train_original_dit(args, logger, device, rank)
        rs_ckpt, rs_losses = train_resshift_dit(args, logger, device, rank)

        if orig_losses and rs_losses:
            save_loss_curves(orig_losses, rs_losses, args.output_dir, logger)
    else:
        logger.info("Skipping training (using provided checkpoints)")
        assert orig_ckpt and rs_ckpt, "Must provide --orig-ckpt and --resshift-ckpt with --skip-training"

    if args.only_train:
        logger.info("--only-train specified, skipping sampling/comparison")
        dist.destroy_process_group()
        return

    # ---- Phase 2: Sampling ----
    logger.info("\n\n>>> PHASE 2: SAMPLING <<<\n")
    assert orig_ckpt and os.path.isfile(orig_ckpt), f"Original DiT checkpoint not found: {orig_ckpt}"
    assert rs_ckpt and os.path.isfile(rs_ckpt), f"ResShift checkpoint not found: {rs_ckpt}"

    orig_raw, orig_svd = sample_original_dit(orig_ckpt, args, logger, device)
    rs_final, rs_xcond = sample_resshift_dit(rs_ckpt, args, logger, device)

    # ---- Phase 3: Comparison ----
    logger.info("\n\n>>> PHASE 3: COMPARISON <<<\n")
    samples_dir = os.path.join(args.output_dir, "samples")
    make_comparison_grid(orig_raw, orig_svd, rs_final, rs_xcond, samples_dir, logger)
    metrics = compute_metrics(orig_svd, rs_final, logger)

    # Save final report
    report = {
        "config": {
            "model": args.model,
            "image_size": args.image_size,
            "num_classes": args.num_classes,
            "train_steps": args.train_steps,
            "n_timestep": args.n_timestep,
            "kappa": args.kappa,
            "svd_ratio": args.svd_ratio,
            "sampling_steps": args.sampling_steps,
            "cfg_scale": args.cfg_scale,
            "class_labels": args.class_labels,
        },
        "metrics": metrics,
    }
    with open(os.path.join(args.output_dir, "comparison_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nAll results saved to {args.output_dir}/")
    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Original DiT vs ResShift DiT")

    # Data & model
    parser.add_argument("--data-path", type=str, default=None, help="Path to ImageFolder dataset")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--pretrained-ckpt", type=str, default=None, help="Pretrained DiT weights for fine-tuning")

    # Training
    parser.add_argument("--train-steps", type=int, default=500, help="Total training steps for each model")
    parser.add_argument("--max-epochs", type=int, default=10, help="Max epochs (will stop early at train-steps)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema", help="VAE for training")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=200)

    # ResShift params
    parser.add_argument("--n-timestep", type=int, default=15)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--svd-ratio", type=float, default=0.25)

    # Sampling
    parser.add_argument("--sampling-steps", type=int, default=250, help="DDPM sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--sample-vae", type=str, choices=["ema", "mse"], default="mse", help="VAE for sampling")
    parser.add_argument("--class-labels", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7],
                        help="Class labels for sampling")

    # Control
    parser.add_argument("--skip-training", action="store_true", help="Skip training, use provided checkpoints")
    parser.add_argument("--only-train", action="store_true", help="Only train, skip sampling/comparison")
    parser.add_argument("--orig-ckpt", type=str, default=None, help="Pretrained original DiT checkpoint")
    parser.add_argument("--resshift-ckpt", type=str, default=None, help="Pretrained ResShift checkpoint")
    parser.add_argument("--output-dir", type=str, default="comparison_results")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
