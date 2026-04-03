"""
Sampling script for DiT with ResShift Diffusion.

Inference pipeline:
  1. Run standard DDPM reverse sampling from T to 0.5T → rough latent z_half
  2. SVD on z_half, keep 25% singular values → x_cond (structural skeleton)
  3. ResShift reverse: starting from noise e_T, iteratively refine the residual
  4. Reconstruct: z_clean = x_cond + e_0
  5. Decode z_clean with VAE → output image

Usage:
    python sample_resshift.py --ckpt results_resshift/000-DiT-XL-2-resshift/checkpoints/0010000.pt
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
import argparse
import os

from models import DiT_models
from diffusion import create_diffusion, create_resshift_diffusion
from diffusers.models import AutoencoderKL
from download import find_model


def svd_lowrank(x, ratio=0.25):
    """SVD low-rank reconstruction keeping top `ratio` fraction of singular values."""
    U, S, Vt = torch.linalg.svd(x)
    r_use = max(int(ratio * S.size(-1)), 1)
    Sr = S[:, :, :r_use]
    recon = (U[:, :, :, :r_use] * Sr.unsqueeze(-2)) @ Vt[:, :, :r_use, :]
    return recon


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint (contains model weights, ema weights, and training args)
    assert args.ckpt is not None, "Must provide --ckpt path"
    state_dict = find_model(args.ckpt)  # handles ema extraction automatically

    # Load training args from checkpoint for model config
    raw_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "args" in raw_ckpt:
        ckpt_args = raw_ckpt["args"]
    else:
        # Fallback: use command-line args for model config
        ckpt_args = args

    # Recreate model with original DiT config (in_channels=4, learn_sigma=True)
    latent_size = getattr(ckpt_args, 'image_size', args.image_size) // 8
    model_name = getattr(ckpt_args, 'model', args.model)
    num_classes = getattr(ckpt_args, 'num_classes', args.num_classes)
    model = DiT_models[model_name](
        input_size=latent_size,
        num_classes=num_classes,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # Create diffusion processes
    n_timestep = getattr(ckpt_args, 'n_timestep', args.n_timestep)
    kappa = getattr(ckpt_args, 'kappa', args.kappa)
    svd_ratio = getattr(ckpt_args, 'svd_ratio', args.svd_ratio)

    ddpm_diffusion = create_diffusion(str(args.ddpm_steps))
    resshift_diffusion = create_resshift_diffusion(
        n_timestep=n_timestep,
        kappa=kappa,
    )

    # VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    print(f"Loaded: model={model_name}, n_timestep={n_timestep}, "
          f"kappa={kappa}, svd_ratio={svd_ratio}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Labels to condition on
    class_labels = [int(x) for x in args.class_labels.split(",")]
    n = len(class_labels)
    y = torch.tensor(class_labels, device=device)

    # === Stage 1: DDPM reverse sampling from T to 0.5T ===
    # Start from pure noise
    z = torch.randn(n, 4, latent_size, latent_size, device=device)

    # For CFG, duplicate z and y
    z_cfg = torch.cat([z, z], 0)
    y_null = torch.tensor([num_classes] * n, device=device)
    y_cfg = torch.cat([y, y_null], 0)
    model_kwargs_ddpm = dict(y=y_cfg, cfg_scale=args.cfg_scale)

    # Run DDPM sampling but only for 50% of total steps (from T to 0.5T)
    z_half = ddpm_diffusion.p_sample_loop(
        model.forward_with_cfg, z_cfg.shape, z_cfg,
        clip_denoised=False, model_kwargs=model_kwargs_ddpm,
        progress=True, device=device,
        clip_point=0.5,  # stop at 50% of steps
    )
    # Take only the conditional half
    z_half = z_half[:n]

    print(f"Stage 1 done: DDPM generated rough latent at 0.5T, shape={z_half.shape}")

    # === Stage 2: SVD low-rank → x_cond ===
    x_cond = svd_lowrank(z_half, ratio=svd_ratio)
    print(f"Stage 2 done: SVD low-rank x_cond, shape={x_cond.shape}")

    # === Stage 3: ResShift reverse sampling ===
    model_kwargs_rs = dict(y=y, x_cond=x_cond)
    z_clean = resshift_diffusion.p_sample_loop(
        model, x_cond,
        clip_denoised=False,
        model_kwargs=model_kwargs_rs,
        device=device, progress=True,
    )
    print(f"Stage 3 done: ResShift refined latent, shape={z_clean.shape}")

    # === Stage 4: VAE decode ===
    samples = vae.decode(z_clean / 0.18215).sample

    # Save results
    save_image(samples, os.path.join(args.output_dir, "resshift_samples.png"),
               nrow=4, normalize=True, value_range=(-1, 1))

    # Also save intermediate x_cond decoded for comparison
    x_cond_decoded = vae.decode(x_cond / 0.18215).sample
    save_image(x_cond_decoded, os.path.join(args.output_dir, "resshift_xcond.png"),
               nrow=4, normalize=True, value_range=(-1, 1))

    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to ResShift DiT checkpoint")
    parser.add_argument("--model", type=str, default="DiT-XL/2", help="DiT model name (fallback if not in ckpt)")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256,
                        help="Image size (fallback if not in ckpt)")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--n-timestep", type=int, default=15, help="ResShift steps (fallback)")
    parser.add_argument("--kappa", type=float, default=1.0, help="ResShift kappa (fallback)")
    parser.add_argument("--svd-ratio", type=float, default=0.25, help="SVD ratio (fallback)")
    parser.add_argument("--output-dir", type=str, default="output_resshift")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--class-labels", type=str, default="207,360,387,974,88,979,417,279",
                        help="Comma-separated class labels for conditional generation")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Classifier-free guidance scale for DDPM stage")
    parser.add_argument("--ddpm-steps", type=int, default=250, help="Total DDPM steps (will run 50%)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
