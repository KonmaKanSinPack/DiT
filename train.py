# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
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
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
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


#################################################################################
#                              FID Evaluation                                   #
#################################################################################

@torch.no_grad()
def evaluate_fid(ema_model, vae, diffusion, fid_data_path, device, logger,
                 num_classes=1000, image_size=256, num_fid_samples=1024,
                 fid_batch_size=8, num_sampling_steps=250, cfg_scale=4.0,
                 use_xcond=True):
    """
    Compute FID score between generated and real images.
    Uses InceptionV3 features (2048-dim) from torchvision.
    Samples with classifier-free guidance (CFG) for quality.
    """
    from scipy import linalg
    from torchvision.models import inception_v3
    import torch.nn.functional as F

    logger.info(f"Computing FID with {num_fid_samples} samples, "
                f"{num_sampling_steps} steps, cfg={cfg_scale}, xcond={use_xcond}...")

    latent_size = image_size // 8

    # Load InceptionV3 for feature extraction
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()
    inception.eval()

    # ImageNet normalization for Inception
    inc_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    inc_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # FID dataset (no random flip for evaluation)
    fid_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    fid_dataset = ImageFolder(fid_data_path, transform=fid_transform)
    fid_loader = DataLoader(fid_dataset, batch_size=fid_batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)

    # Create diffusion with fewer steps for faster sampling
    sample_diffusion = create_diffusion(str(num_sampling_steps))

    real_feats = []
    fake_feats = []
    n_collected = 0

    for x_batch, y_batch in fid_loader:
        if n_collected >= num_fid_samples:
            break

        bs = x_batch.shape[0]
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # --- Real image inception features ---
        real_01 = (x_batch + 1) / 2  # [-1,1] -> [0,1]
        real_299 = F.interpolate(real_01, size=(299, 299), mode='bilinear', align_corners=False)
        real_299 = (real_299 - inc_mean) / inc_std
        real_feats.append(inception(real_299).cpu().numpy())

        # --- Generate fake images with CFG ---
        # Encode to latent space
        latents = vae.encode(x_batch).latent_dist.sample().mul_(0.18215)

        # Prepare x_cond if needed (same as training: noise at half_time + SVD)
        x_cond_cfg = None
        if use_xcond:
            half_time = int(0.5 * diffusion.num_timesteps) * torch.ones(
                (bs,), dtype=torch.int, device=device
            )
            noisy = diffusion.q_sample(latents, half_time)
            U, S, Vt = torch.linalg.svd(noisy)
            r_use = int(0.25 * S.size(-1))
            Sr = S[:, :, :r_use]
            x_cond = (U[:, :, :, :r_use] * Sr.unsqueeze(-2)) @ Vt[:, :, :r_use, :]
            # Double x_cond for CFG (conditional + unconditional use same x_cond)
            x_cond_cfg = torch.cat([x_cond, x_cond], 0)

        # Prepare CFG inputs: double z and y
        z = torch.randn(bs, 4, latent_size, latent_size, device=device)
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([num_classes] * bs, device=device)
        y_cfg = torch.cat([y_batch, y_null], 0)

        model_kwargs = dict(y=y_cfg, cfg_scale=cfg_scale)
        if use_xcond:
            model_kwargs["x_cond"] = x_cond_cfg

        samples = sample_diffusion.p_sample_loop(
            ema_model.forward_with_cfg, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=False, device=device
        )
        # Remove the unconditional half
        samples, _ = samples.chunk(2, dim=0)

        # Decode to pixel space
        fake_pixels = vae.decode(samples / 0.18215).sample
        fake_01 = ((fake_pixels + 1) / 2).clamp(0, 1)
        fake_299 = F.interpolate(fake_01, size=(299, 299), mode='bilinear', align_corners=False)
        fake_299 = (fake_299 - inc_mean) / inc_std
        fake_feats.append(inception(fake_299).cpu().numpy())

        n_collected += bs
        if n_collected % (fid_batch_size * 8) == 0:
            logger.info(f"  FID progress: {n_collected}/{num_fid_samples}")

    # Concatenate and trim to exact count
    real_feats = np.concatenate(real_feats, axis=0)[:num_fid_samples]
    fake_feats = np.concatenate(fake_feats, axis=0)[:num_fid_samples]

    # Compute statistics
    mu_r = np.mean(real_feats, axis=0)
    sigma_r = np.cov(real_feats, rowvar=False)
    mu_f = np.mean(fake_feats, axis=0)
    sigma_f = np.cov(fake_feats, rowvar=False)

    # Compute FID
    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))

    del inception
    torch.cuda.empty_cache()

    logger.info(f"FID Score: {fid:.4f} ({n_collected} samples)")
    return fid


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Load pretrained checkpoint if provided
    if args.ckpt:
        state_dict = find_model(args.ckpt)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint from {args.ckpt}")
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
           
            #----执行svd分解
            half_time = int(0.5*diffusion.num_timesteps)*torch.ones((x.shape[0], ),dtype=torch.int, device=device)
            recon = diffusion.q_sample(x, half_time)  # Add noise to the latents according to the diffusion process
            U, S, Vt = torch.linalg.svd(recon)

            # energy = S ** 2
            # cumulative_energy = torch.cumsum(energy, dim=-1)
            # total_energy = energy.sum(dim=-1, keepdim=True)
            
            # energy_threshold = 0.9
            # mask = (cumulative_energy / total_energy) <= energy_threshold
            # r_use = mask.sum(dim=-1).max().item() # 取全局最大的 r 以保持张量对齐
            # r_use = max(int(r_use), 1) # 至少保留一个奇异值
            r_use = int(0.25*S.size(-1))

            Sr =S[:, :, :r_use]
            recon = (U[:, :, :, :r_use] * Sr.unsqueeze(-2)) @ Vt[:, :, :r_use, :]#.reshape(b, c, h, w)
            # print(f"recon shape:{recon.shape}")
            #----结束svd分解

            model_kwargs = dict(y=y,x_cond=recon)  
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        # FID evaluation at end of epoch
        if args.fid_every > 0 and (epoch + 1) % args.fid_every == 0:
            if rank == 0:
                logger.info(f"Epoch {epoch}: running FID evaluation...")
                fid_score = evaluate_fid(
                    ema_model=ema,
                    vae=vae,
                    diffusion=diffusion,
                    fid_data_path=args.fid_data_path,
                    device=device,
                    logger=logger,
                    num_classes=args.num_classes,
                    image_size=args.image_size,
                    num_fid_samples=args.fid_samples,
                    fid_batch_size=args.fid_batch_size,
                    num_sampling_steps=args.fid_sampling_steps,
                    cfg_scale=args.fid_cfg_scale,
                    use_xcond=args.fid_use_xcond,
                )
            dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--fid-every", type=int, default=0, help="Evaluate FID every N epochs (0=disabled)")
    parser.add_argument("--fid-data-path", type=str, default="./imagenet100/train", help="Path to FID evaluation data")
    parser.add_argument("--fid-samples", type=int, default=1024, help="Number of samples for FID")
    parser.add_argument("--fid-batch-size", type=int, default=8, help="Batch size for FID evaluation")
    parser.add_argument("--fid-sampling-steps", type=int, default=250, help="Diffusion sampling steps for FID")
    parser.add_argument("--fid-cfg-scale", type=float, default=4.0, help="CFG scale for FID sampling")
    parser.add_argument("--fid-use-xcond", action="store_true", help="Use x_cond (SVD) during FID evaluation")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to pretrained DiT checkpoint")
    args = parser.parse_args()
    main(args)
