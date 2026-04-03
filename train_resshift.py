"""
Training script for DiT with ResShift Diffusion.

Pipeline:
  1. Encode clean image with VAE → latent z (4 channels)
  2. DDPM forward process at t=0.5T → noisy latent z_noisy
  3. SVD on z_noisy, keep 25% singular values → x_cond (low-rank structural skeleton)
  4. ResShift learns the residual e_0 = z_clean - x_cond
  5. Model input: (e_t + x_cond), model output: predicted e_0

This uses the original DiT architecture with learn_sigma=True and in_channels=4,
so pretrained DiT weights can be loaded directly for fine-tuning.

Usage (single-GPU):
    torchrun --nnodes=1 --nproc_per_node=1 train_resshift.py --data-path /path/to/imagenet

Usage (multi-GPU DDP):
    torchrun --nnodes=1 --nproc_per_node=N train_resshift.py --data-path /path/to/imagenet
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
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion, create_resshift_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from fid_utils import compute_fid_score


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


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
    """
    Perform SVD on a [B, C, H, W] tensor and reconstruct with only
    the top `ratio` fraction of singular values.
    """
    U, S, Vt = torch.linalg.svd(x)
    r_use = max(int(ratio * S.size(-1)), 1)
    Sr = S[:, :, :r_use]
    recon = (U[:, :, :, :r_use] * Sr.unsqueeze(-2)) @ Vt[:, :, :r_use, :]
    return recon


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training requires at least one GPU."

    # Setup DDP
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-resshift"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model — use original DiT architecture (in_channels=4, learn_sigma=True)
    # so that pretrained DiT weights can be loaded
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    )  # default: in_channels=4, learn_sigma=True

    # Load pretrained DiT weights for fine-tuning via find_model()
    # Supports: auto-download ("DiT-XL-2-256x256.pt"), train.py checkpoints, or direct state_dict
    if args.pretrained_ckpt:
        state_dict = find_model(args.pretrained_ckpt)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded pretrained weights from {args.pretrained_ckpt}")

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])

    # Create standard DDPM (for generating x_cond via q_sample at 0.5T + SVD)
    ddpm_diffusion = create_diffusion(timestep_respacing="")  # 1000 steps, linear

    # Create ResShift diffusion (for the residual refinement stage)
    resshift_diffusion = create_resshift_diffusion(
        n_timestep=args.n_timestep,
        kappa=args.kappa,
    )

    # VAE for encoding images to latent space
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"ResShift: n_timestep={args.n_timestep}, kappa={args.kappa}")
    logger.info(f"DDPM half_T={ddpm_diffusion.num_timesteps // 2}, SVD ratio={args.svd_ratio}")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Dataset: plain images normalized to [-1, 1]
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=rank,
        shuffle=True, seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare training
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    best_fid = float('inf')

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y_label in loader:
            x = x.to(device)
            y_label = y_label.to(device)

            with torch.no_grad():
                # Step 1: Encode to VAE latent space (4 channels)
                z_clean = vae.encode(x).latent_dist.sample().mul_(0.18215)

                # Step 2: DDPM forward at t = 0.5T → noisy latent
                half_T = int(0.5 * ddpm_diffusion.num_timesteps)
                t_half = torch.full((z_clean.shape[0],), half_T, dtype=torch.long, device=device)
                z_noisy = ddpm_diffusion.q_sample(z_clean, t_half)

                # Step 3: SVD low-rank approximation → x_cond
                x_cond = svd_lowrank(z_noisy, ratio=args.svd_ratio)

            # Random ResShift timesteps
            t = torch.randint(0, resshift_diffusion.num_timesteps, (z_clean.shape[0],), device=device)

            # model_kwargs: class labels + structural x_cond
            model_kwargs = dict(y=y_label, x_cond=x_cond)

            # Step 4: ResShift training loss
            # x_start = z_clean (clean latent), y = x_cond (degraded via DDPM+SVD)
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

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    # Keep only latest periodic checkpoint + best.pt
                    for old_ckpt in sorted(glob(f"{checkpoint_dir}/0*.pt")):
                        if old_ckpt != checkpoint_path:
                            os.remove(old_ckpt)
                            logger.info(f"Removed old checkpoint: {old_ckpt}")
                dist.barrier()

            # Periodic FID evaluation
            if args.fid_every > 0 and train_steps % args.fid_every == 0 and train_steps > 0:
                if rank == 0:
                    logger.info(f"Computing FID at step {train_steps} ({args.fid_samples} samples)...")
                    fid_score = compute_fid_score(
                        ema, vae, args.data_path, device,
                        num_samples=args.fid_samples, mode="resshift",
                        cfg_scale=4.0, num_classes=args.num_classes,
                        latent_size=latent_size, batch_size=16,
                        tmp_dir=f"{experiment_dir}/fid_tmp",
                        ddpm_steps=250, n_timestep=args.n_timestep,
                        kappa=args.kappa, svd_ratio=args.svd_ratio,
                    )
                    logger.info(f"FID at step {train_steps}: {fid_score:.2f} (best: {best_fid:.2f})")
                    if fid_score < best_fid:
                        best_fid = fid_score
                        best_path = f"{checkpoint_dir}/best.pt"
                        best_ckpt = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args,
                            "fid": fid_score,
                            "step": train_steps,
                        }
                        torch.save(best_ckpt, best_path)
                        logger.info(f"New best FID! Saved best checkpoint to {best_path}")
                    model.train()
                dist.barrier()

    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results_resshift")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--n-timestep", type=int, default=15, help="Number of ResShift diffusion steps")
    parser.add_argument("--kappa", type=float, default=1.0, help="ResShift kappa parameter")
    parser.add_argument("--svd-ratio", type=float, default=0.25, help="Fraction of singular values to keep")
    parser.add_argument("--pretrained-ckpt", type=str, default=None,
                        help="Path to pretrained DiT checkpoint for fine-tuning")
    parser.add_argument("--fid-every", type=int, default=0,
                        help="Compute FID every N steps (0=disabled)")
    parser.add_argument("--fid-samples", type=int, default=1024,
                        help="Number of samples for FID evaluation")
    args = parser.parse_args()
    main(args)
