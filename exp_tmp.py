import torch
from PIL import Image
import numpy as np
from diffusion import create_diffusion
from torchvision.utils import save_image
if __name__ == "__main__":
    img = Image.open('hutao.jpg')
    img_tensor = torch.from_numpy(np.array(img)).float().cpu()
    x = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0  # Convert to CxHxW and add batch dimension, normalize to [0, 1]
    diffusion = create_diffusion(timestep_respacing="") 

     #----执行svd分解
    t = int(0.5*diffusion.num_timesteps)*torch.ones((x.shape[0], ),dtype=torch.int)
    recon = diffusion.q_sample(x, t)  # Add noise to the latents according to the diffusion process
    save_image(recon, 'recon1.png')
    U, S, Vt = torch.linalg.svd(recon)
    print(f"U shape:{U.shape}, S shape:{S.shape}, Vt shape:{Vt.shape}")
    print(f"recon shape:{recon.shape}")

    energy = S ** 2
    cumulative_energy = torch.cumsum(energy, dim=-1)
    total_energy = energy.sum(dim=-1, keepdim=True)
    
    energy_threshold = 0.9
    mask = (cumulative_energy / total_energy) <= energy_threshold
    r_use = mask.sum(dim=-1).max().item() # 取全局最大的 r 以保持张量对齐
    r_use = max(int(r_use), 1) # 至少保留一个奇异值
    print(f"保留的奇异值数量: {r_use}")
    r_use = int(0.25*S.size(-1))
    print(f"保留的奇异值数量: {r_use}")

    Sr =S[:, :, :r_use]
    recon = (U[:, :, :, :r_use] * Sr.unsqueeze(-2)) @ Vt[:, :, :r_use, :]#.reshape(b, c, h, w)
    # print(f"recon shape:{recon.shape}")
    #----结束svd分解
    save_image(recon, 'recon2.png')
    print(img_tensor.shape)
