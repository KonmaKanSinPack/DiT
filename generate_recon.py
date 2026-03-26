from PIL import Image
import torch
import numpy as np
from torchvision.utils import save_image
import torchvision
image = Image.open("hutao.jpg").convert("RGB")
image = image.resize((512, 512))
image_np = np.array(image).astype(np.float32) / 255.0
image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, H, W)
print(f"Original image shape: {image_tensor.shape}")

U, S, Vt = torch.linalg.svd(image_tensor)
print(f"U shape: {U.shape}, S shape: {S.shape}, Vt shape: {Vt.shape}")
energy = S ** 2
cumulative_energy = torch.cumsum(energy, dim=-1)
total_energy = energy.sum(dim=-1, keepdim=True)

energy_threshold = 0.9
mask = (cumulative_energy / total_energy) <= energy_threshold
r_use = mask.sum(dim=-1).max().item() # 取全局最大的 r 以保持张量对齐
r_use = max(int(r_use), 1) # 至少保留一个奇异值

r_use = 16
print(r_use)
Sr =S[:, :, :r_use]
recon = (U[:, :, :, :r_use] * Sr.unsqueeze(-2)) @ Vt[:, :, :r_use, :]#.reshape(b, c, h, w)
# recon = image_tensor
save_image(recon, "recon.png")
            