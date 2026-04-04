"""
Quick standalone FID test script - loads pretrained model, runs FID evaluation.
No DDP, no training. For verifying FID evaluation correctness.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import numpy as np
import argparse
import logging

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


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


@torch.no_grad()
def main(args):
    logging.basicConfig(level=logging.INFO, format='[\033[34m%(asctime)s\033[0m] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from scipy import linalg
    from torchvision.models import inception_v3
    import torch.nn.functional as F

    # Load model
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info(f"Loaded DiT from {args.ckpt}, params: {sum(p.numel() for p in model.parameters()):,}")

    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    sample_diffusion = create_diffusion(str(args.num_sampling_steps))

    # Load InceptionV3
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()
    inception.eval()
    inc_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    inc_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Dataset
    fid_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    fid_dataset = ImageFolder(args.data_path, transform=fid_transform)
    fid_loader = DataLoader(fid_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)

    logger.info(f"FID test: {args.num_samples} samples, {args.num_sampling_steps} steps, "
                f"cfg={args.cfg_scale}, use_xcond={args.use_xcond}")

    real_feats, fake_feats = [], []
    n_collected = 0

    for x_batch, y_batch in fid_loader:
        if n_collected >= args.num_samples:
            break

        bs = x_batch.shape[0]
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Real features
        real_01 = (x_batch + 1) / 2
        real_299 = F.interpolate(real_01, size=(299, 299), mode='bilinear', align_corners=False)
        real_299 = (real_299 - inc_mean) / inc_std
        real_feats.append(inception(real_299).cpu().numpy())

        # Generate with CFG
        latents = vae.encode(x_batch).latent_dist.sample().mul_(0.18215)

        x_cond_cfg = None
        if args.use_xcond:
            half_time = int(0.5 * diffusion.num_timesteps) * torch.ones((bs,), dtype=torch.int, device=device)
            noisy = diffusion.q_sample(latents, half_time)
            U, S, Vt = torch.linalg.svd(noisy)
            r_use = int(0.25 * S.size(-1))
            Sr = S[:, :, :r_use]
            x_cond = (U[:, :, :, :r_use] * Sr.unsqueeze(-2)) @ Vt[:, :, :r_use, :]
            x_cond_cfg = torch.cat([x_cond, x_cond], 0)

        z = torch.randn(bs, 4, latent_size, latent_size, device=device)
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([args.num_classes] * bs, device=device)
        y_cfg = torch.cat([y_batch, y_null], 0)

        model_kwargs = dict(y=y_cfg, cfg_scale=args.cfg_scale)
        if args.use_xcond:
            model_kwargs["x_cond"] = x_cond_cfg

        samples = sample_diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=False, device=device
        )
        samples, _ = samples.chunk(2, dim=0)

        fake_pixels = vae.decode(samples / 0.18215).sample
        fake_01 = ((fake_pixels + 1) / 2).clamp(0, 1)
        fake_299 = F.interpolate(fake_01, size=(299, 299), mode='bilinear', align_corners=False)
        fake_299 = (fake_299 - inc_mean) / inc_std
        fake_feats.append(inception(fake_299).cpu().numpy())

        n_collected += bs
        logger.info(f"  Progress: {n_collected}/{args.num_samples}")

    real_feats = np.concatenate(real_feats, axis=0)[:args.num_samples]
    fake_feats = np.concatenate(fake_feats, axis=0)[:args.num_samples]

    mu_r, sigma_r = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)
    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))

    logger.info(f"=== FID Score: {fid:.4f} ({n_collected} samples) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="pretrained_models/DiT-XL-2-256x256.pt")
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--data-path", type=str, default="./imagenet100/train")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, default="ema")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--use-xcond", action="store_true")
    args = parser.parse_args()
    main(args)
