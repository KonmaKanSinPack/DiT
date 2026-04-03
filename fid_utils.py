"""
FID evaluation utilities for DiT training.
Generates samples and computes FID against real image directory.
Supports both standard DDPM and ResShift sampling pipelines.
"""
import os
import torch
import numpy as np
from torchvision.utils import save_image
from PIL import Image
from cleanfid import fid
from diffusion import create_diffusion, create_resshift_diffusion


def svd_lowrank(x, ratio=0.25):
    U, S, Vt = torch.linalg.svd(x)
    r_use = max(int(ratio * S.size(-1)), 1)
    Sr = S[:, :, :r_use]
    recon = (U[:, :, :, :r_use] * Sr.unsqueeze(-2)) @ Vt[:, :, :r_use, :]
    return recon


@torch.no_grad()
def generate_samples_ddpm(model, vae, diffusion, num_samples, num_classes,
                          latent_size, device, cfg_scale=4.0, batch_size=16):
    """Generate samples using standard DDPM reverse sampling (for original DiT)."""
    model.eval()
    all_samples = []
    num_generated = 0
    while num_generated < num_samples:
        bs = min(batch_size, num_samples - num_generated)
        z = torch.randn(bs, 4, latent_size, latent_size, device=device)
        y = torch.randint(0, num_classes, (bs,), device=device)

        # CFG setup
        z_cfg = torch.cat([z, z], 0)
        y_null = torch.tensor([num_classes] * bs, device=device)
        y_cfg = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y_cfg, cfg_scale=cfg_scale)

        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z_cfg.shape, z_cfg,
            clip_denoised=False, model_kwargs=model_kwargs,
            progress=False, device=device,
        )
        samples = samples[:bs]
        samples = vae.decode(samples / 0.18215).sample
        all_samples.append(samples.cpu())
        num_generated += bs

    return torch.cat(all_samples, dim=0)[:num_samples]


@torch.no_grad()
def generate_samples_resshift(model, vae, num_samples, num_classes,
                              latent_size, device, cfg_scale=4.0,
                              ddpm_steps=250, n_timestep=15, kappa=1.0,
                              svd_ratio=0.25, batch_size=16):
    """Generate samples using DDPM→SVD→ResShift pipeline (for ResShift DiT)."""
    model.eval()
    ddpm_diffusion = create_diffusion(str(ddpm_steps))
    resshift_diffusion = create_resshift_diffusion(n_timestep=n_timestep, kappa=kappa)

    all_samples = []
    num_generated = 0
    while num_generated < num_samples:
        bs = min(batch_size, num_samples - num_generated)
        z = torch.randn(bs, 4, latent_size, latent_size, device=device)
        y = torch.randint(0, num_classes, (bs,), device=device)

        # Stage 1: DDPM reverse sampling to 0.5T
        z_cfg = torch.cat([z, z], 0)
        y_null = torch.tensor([num_classes] * bs, device=device)
        y_cfg = torch.cat([y, y_null], 0)
        model_kwargs_ddpm = dict(y=y_cfg, cfg_scale=cfg_scale)

        z_half = ddpm_diffusion.p_sample_loop(
            model.forward_with_cfg, z_cfg.shape, z_cfg,
            clip_denoised=False, model_kwargs=model_kwargs_ddpm,
            progress=False, device=device,
            clip_point=0.5,
        )
        z_half = z_half[:bs]

        # Stage 2: SVD low-rank
        x_cond = svd_lowrank(z_half, ratio=svd_ratio)

        # Stage 3: ResShift reverse sampling
        model_kwargs_rs = dict(y=y, x_cond=x_cond)
        z_clean = resshift_diffusion.p_sample_loop(
            model, x_cond,
            clip_denoised=False,
            model_kwargs=model_kwargs_rs,
            device=device, progress=False,
        )

        # Stage 4: VAE decode
        samples = vae.decode(z_clean / 0.18215).sample
        all_samples.append(samples.cpu())
        num_generated += bs

    return torch.cat(all_samples, dim=0)[:num_samples]


def save_samples_to_dir(samples, output_dir):
    """Save a batch of samples (torch tensor, range [-1,1]) as individual PNG files."""
    os.makedirs(output_dir, exist_ok=True)
    samples = (samples.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
    samples = (samples * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    for i, img_np in enumerate(samples):
        Image.fromarray(img_np).save(os.path.join(output_dir, f"{i:05d}.png"))


def compute_fid_score(model, vae, real_images_dir, device, num_samples=1024,
                      mode="ddpm", cfg_scale=4.0, num_classes=1000,
                      latent_size=32, batch_size=16, tmp_dir="/tmp/fid_gen",
                      ddpm_steps=250, n_timestep=15, kappa=1.0, svd_ratio=0.25):
    """
    Generate samples and compute FID against a directory of real images.

    Args:
        mode: "ddpm" for standard DDPM sampling, "resshift" for DDPM→SVD→ResShift
    Returns:
        FID score (float)
    """
    # Generate samples
    if mode == "ddpm":
        diffusion = create_diffusion(str(ddpm_steps))
        samples = generate_samples_ddpm(
            model, vae, diffusion, num_samples, num_classes,
            latent_size, device, cfg_scale=cfg_scale, batch_size=batch_size
        )
    elif mode == "resshift":
        samples = generate_samples_resshift(
            model, vae, num_samples, num_classes, latent_size, device,
            cfg_scale=cfg_scale, ddpm_steps=ddpm_steps, n_timestep=n_timestep,
            kappa=kappa, svd_ratio=svd_ratio, batch_size=batch_size
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Save generated samples
    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    save_samples_to_dir(samples, tmp_dir)

    # Compute FID
    fid_score = fid.compute_fid(tmp_dir, real_images_dir,
                                mode="clean", num_workers=4)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return fid_score
