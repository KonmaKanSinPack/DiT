"""
End-to-end test for ResShift Diffusion with DiT.

Pipeline: DDPM q_sample at 0.5T → SVD (25% singular values) → x_cond → ResShift refinement.
Uses original DiT architecture (in_channels=4, learn_sigma=True) to be compatible
with pretrained weights. No VAE needed for testing — uses synthetic 4-channel latents.

Usage:
    cd DiT && python test_resshift.py
"""
import torch
import argparse
import os
import sys

from models import DiT_models
from diffusion import create_diffusion, create_resshift_diffusion
from diffusion.gaussian_diffusion import (
    make_resshift_sqrt_etas_schedule, ResShiftDiffusion, _extract_into_tensor
)


def svd_lowrank(x, ratio=0.25):
    """SVD low-rank reconstruction keeping top `ratio` fraction of singular values."""
    U, S, Vt = torch.linalg.svd(x)
    r_use = max(int(ratio * S.size(-1)), 1)
    Sr = S[:, :, :r_use]
    recon = (U[:, :, :, :r_use] * Sr.unsqueeze(-2)) @ Vt[:, :, :r_use, :]
    return recon


def test_schedule():
    """Test that the ResShift schedule is correctly computed."""
    print("=" * 60)
    print("[1/6] Testing ResShift schedule...")
    sqrt_etas = make_resshift_sqrt_etas_schedule(n_timestep=15)
    assert len(sqrt_etas) == 15, f"Expected 15 timesteps, got {len(sqrt_etas)}"
    assert sqrt_etas[0] < sqrt_etas[-1], "Schedule should be monotonically increasing"
    assert sqrt_etas[-1] ** 2 <= 1.0, "Final eta should be <= 1.0"
    print(f"  sqrt_etas range: [{sqrt_etas[0]:.6f}, {sqrt_etas[-1]:.6f}]")
    print(f"  etas range: [{sqrt_etas[0]**2:.6f}, {sqrt_etas[-1]**2:.6f}]")
    print("  PASSED\n")


def test_ddpm_svd_degradation():
    """Test the DDPM→SVD degradation pipeline that produces x_cond."""
    print("=" * 60)
    print("[2/6] Testing DDPM q_sample at 0.5T + SVD degradation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ddpm = create_diffusion(timestep_respacing="")  # 1000 steps
    B, C, H, W = 2, 4, 32, 32  # 4-channel latent
    z_clean = torch.randn(B, C, H, W, device=device)

    # DDPM forward at t=0.5T
    half_T = ddpm.num_timesteps // 2
    t_half = torch.full((B,), half_T, dtype=torch.long, device=device)
    z_noisy = ddpm.q_sample(z_clean, t_half)
    assert z_noisy.shape == z_clean.shape
    print(f"  DDPM q_sample at t={half_T}: z_noisy std = {z_noisy.std().item():.4f}")

    # SVD low-rank
    x_cond = svd_lowrank(z_noisy, ratio=0.25)
    assert x_cond.shape == z_clean.shape
    # x_cond should be a low-rank approximation (less information)
    svd_diff = (z_noisy - x_cond).abs().mean().item()
    print(f"  SVD 25%: mean diff from z_noisy = {svd_diff:.4f} (should be > 0)")
    assert svd_diff > 0, "SVD should discard information"
    print("  PASSED\n")


def test_forward_process():
    """Test the ResShift forward (q_sample) process."""
    print("=" * 60)
    print("[3/6] Testing ResShift forward process (q_sample)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion = create_resshift_diffusion(n_timestep=15, kappa=1.0)

    B, C, H, W = 2, 4, 32, 32  # 4-channel latent
    e_0 = torch.randn(B, C, H, W, device=device)

    # t=0: should be close to e_0 (minimal noise)
    t0 = torch.zeros(B, dtype=torch.long, device=device)
    e_t0 = diffusion.q_sample(e_0, t0, noise=torch.zeros_like(e_0))
    eta_0 = diffusion.etas[0]
    expected = (1 - eta_0) * e_0
    diff = (e_t0 - expected).abs().max().item()
    assert diff < 1e-5, f"At t=0 with zero noise, diff={diff}"
    print(f"  t=0 (no noise): max diff from expected = {diff:.2e} -- OK")

    # t=T-1: should be mostly noise
    t_last = torch.full((B,), diffusion.num_timesteps - 1, dtype=torch.long, device=device)
    e_tT = diffusion.q_sample(e_0, t_last)
    assert e_tT.shape == e_0.shape
    print(f"  t=T-1: e_t std = {e_tT.std().item():.4f}")
    print("  PASSED\n")


def test_training():
    """Test training with the full DDPM→SVD→ResShift pipeline using synthetic data."""
    print("=" * 60)
    print("[4/6] Testing training loop (DDPM→SVD→ResShift, 5 steps)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use DiT-S/8 with original params: in_channels=4, learn_sigma=True
    latent_size = 32  # e.g. 256/8=32
    model = DiT_models["DiT-S/8"](
        input_size=latent_size,
        num_classes=10,
    ).to(device)  # default: in_channels=4, learn_sigma=True
    model.train()

    ddpm = create_diffusion(timestep_respacing="")
    resshift = create_resshift_diffusion(n_timestep=15, kappa=1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    B = 2

    losses = []
    for step in range(5):
        # Synthetic clean latents (as if from VAE encoder)
        z_clean = torch.randn(B, 4, latent_size, latent_size, device=device)
        y_label = torch.randint(0, 10, (B,), device=device)

        with torch.no_grad():
            # DDPM forward at 0.5T
            half_T = ddpm.num_timesteps // 2
            t_half = torch.full((B,), half_T, dtype=torch.long, device=device)
            z_noisy = ddpm.q_sample(z_clean, t_half)
            # SVD low-rank → x_cond
            x_cond = svd_lowrank(z_noisy, ratio=0.25)

        # Random ResShift timesteps
        t = torch.randint(0, resshift.num_timesteps, (B,), device=device)

        model_kwargs = dict(y=y_label, x_cond=x_cond)
        loss_dict = resshift.training_losses(
            model, x_start=z_clean, y=x_cond, t=t, model_kwargs=model_kwargs,
        )
        loss = loss_dict["loss"].mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"  Step {step+1}: loss = {loss.item():.6f}")

    assert all(l > 0 for l in losses), "All losses should be positive"
    print("  PASSED\n")
    return model, resshift, ddpm


def test_sampling(model, resshift, ddpm):
    """Test the full inference pipeline: DDPM partial → SVD → ResShift."""
    print("=" * 60)
    print("[5/6] Testing sampling loop (full inference pipeline)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    B = 2
    latent_size = 32

    with torch.no_grad():
        # Simulate: create x_cond from a ground-truth latent
        z_clean = torch.randn(B, 4, latent_size, latent_size, device=device)
        half_T = ddpm.num_timesteps // 2
        t_half = torch.full((B,), half_T, dtype=torch.long, device=device)
        z_noisy = ddpm.q_sample(z_clean, t_half)
        x_cond = svd_lowrank(z_noisy, ratio=0.25)

        y_label = torch.zeros(B, dtype=torch.long, device=device)
        model_kwargs = dict(y=y_label, x_cond=x_cond)

        # ResShift reverse sampling
        z_reconstructed = resshift.p_sample_loop(
            model, x_cond, clip_denoised=False,
            model_kwargs=model_kwargs, device=device, progress=True,
        )

    assert z_reconstructed.shape == (B, 4, latent_size, latent_size), \
        f"Expected shape {(B, 4, latent_size, latent_size)}, got {z_reconstructed.shape}"
    print(f"  Output shape: {z_reconstructed.shape}")
    print(f"  Output range: [{z_reconstructed.min().item():.4f}, {z_reconstructed.max().item():.4f}]")

    diff = (z_reconstructed - x_cond).abs().mean().item()
    print(f"  Mean diff from x_cond: {diff:.6f}")
    print("  PASSED\n")


def test_save_load(model, resshift, ddpm):
    """Test checkpoint save and load cycle."""
    print("=" * 60)
    print("[6/6] Testing checkpoint save/load...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulate saving
    ckpt_path = "/tmp/test_resshift_ckpt.pt"
    fake_args = argparse.Namespace(
        model="DiT-S/8", image_size=256, num_classes=10,
        n_timestep=15, kappa=1.0, svd_ratio=0.25, vae="ema",
    )
    checkpoint = {
        "model": model.state_dict(),
        "ema": model.state_dict(),
        "args": fake_args,
    }
    torch.save(checkpoint, ckpt_path)

    # Load
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    loaded_args = ckpt["args"]
    model2 = DiT_models[loaded_args.model](
        input_size=loaded_args.image_size // 8,
        num_classes=loaded_args.num_classes,
    ).to(device)
    model2.load_state_dict(ckpt["ema"])
    model2.eval()

    resshift2 = create_resshift_diffusion(
        n_timestep=loaded_args.n_timestep,
        kappa=loaded_args.kappa,
    )

    # Check outputs match
    B = 2
    latent_size = loaded_args.image_size // 8
    z_clean = torch.randn(B, 4, latent_size, latent_size, device=device)
    half_T = ddpm.num_timesteps // 2
    t_half = torch.full((B,), half_T, dtype=torch.long, device=device)

    with torch.no_grad():
        z_noisy = ddpm.q_sample(z_clean, t_half)
        x_cond = svd_lowrank(z_noisy, ratio=0.25)
        y_label = torch.zeros(B, dtype=torch.long, device=device)
        model_kwargs = dict(y=y_label, x_cond=x_cond)

        torch.manual_seed(42)
        out1 = resshift.p_sample_loop(model, x_cond, model_kwargs=model_kwargs, device=device)
        torch.manual_seed(42)
        out2 = resshift2.p_sample_loop(model2, x_cond, model_kwargs=model_kwargs, device=device)

    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-4, f"Loaded model output differs: max diff = {diff}"
    print(f"  Max diff between original and loaded model: {diff:.2e}")
    os.remove(ckpt_path)
    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ResShift Diffusion + DiT — End-to-End Test Suite")
    print("  Pipeline: DDPM → SVD → ResShift (VAE latent space)")
    print("=" * 60 + "\n")

    test_schedule()
    test_ddpm_svd_degradation()
    test_forward_process()
    model, resshift, ddpm = test_training()
    test_sampling(model, resshift, ddpm)
    test_save_load(model, resshift, ddpm)

    print("=" * 60)
    print("  ALL TESTS PASSED!")
    print("=" * 60)
