# ResShift Diffusion + DiT 说明文档

## 1. 概述

本项目在 DiT (Diffusion Transformer) 框架中实现了 **DDPM → SVD → ResShift** 三阶段生成管线。
生成流程为：先用标准 DDPM 在 0.5T 处生成粗略图像，再通过 SVD 分解保留 25% 奇异值得到结构骨架 `x_cond`，
最后用 ResShift 残差偏移扩散对残差进行精细化恢复。

**关键特性**：完全兼容原始 DiT 预训练权重（`in_channels=4`, `learn_sigma=True`），
工作在 VAE 隐空间中（4 通道），支持直接加载 DiT-XL/2 等官方权重进行微调。

---

## 2. 整体流程

### 2.1 三阶段管线

```
┌───────────────────────────────────────────────────────────────────────┐
│                        训练流程                                        │
│                                                                       │
│  原始图像 ──VAE编码──→ z_clean (4ch, 32×32)                            │
│                           │                                           │
│                    DDPM q_sample(t=0.5T)                              │
│                           │                                           │
│                      z_noisy (添加噪声)                                │
│                           │                                           │
│                    SVD 保留 25% 奇异值                                  │
│                           │                                           │
│                       x_cond (低秩骨架)           e_0 = z_clean - x_cond│
│                           │                         │                 │
│                           └─────┬───────────────────┘                 │
│                                 │                                     │
│                    ResShift 前向加噪 e_t                                │
│                                 │                                     │
│                    模型输入: (e_t + x_cond, t)                          │
│                    模型输出: 预测 ê_0                                   │
│                    损失: MSE(ê_0, e_0)                                 │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│                        推理流程                                        │
│                                                                       │
│  阶段1: DDPM 反向采样 (T → 0.5T)                                       │
│         从纯噪声开始，运行 DDPM 的前 50% 步                              │
│         得到粗略隐向量 z_half                                           │
│                                                                       │
│  阶段2: SVD 低秩近似                                                    │
│         对 z_half 进行 SVD 分解，保留 25% 奇异值                          │
│         得到结构骨架 x_cond                                             │
│                                                                       │
│  阶段3: ResShift 反向采样 (仅 15 步)                                     │
│         从噪声 e_T 开始，迭代恢复残差 e_0                                │
│         z_clean = x_cond + e_0                                        │
│                                                                       │
│  阶段4: VAE 解码                                                       │
│         z_clean ──VAE解码──→ 最终输出图像                                │
└───────────────────────────────────────────────────────────────────────┘
```

### 2.2 各阶段详解

**阶段 1：DDPM 生成粗略结构**
- 标准 DDPM 扩散过程（1000步线性 beta schedule）
- 训练时：对干净隐向量 `z_clean` 用 `q_sample` 在 `t=500` 处加噪 → `z_noisy`
- 推理时：从纯高斯噪声开始，用 `p_sample_loop` 反向采样，但只运行 50% 的步数（使用 `clip_point=0.5`）

**阶段 2：SVD 低秩分解**
- 对 `z_noisy`/`z_half` 做 SVD：`U, S, Vt = torch.linalg.svd(z_noisy)`
- 保留前 25% 的奇异值：`r_use = int(0.25 * S.size(-1))`
- 重建低秩近似：`x_cond = U[:,:,:,:r] * S[:,:,:r] @ Vt[:,:,:r,:]`
- 效果：去除高频细节，保留主要结构信息

**阶段 3：ResShift 残差精细化（15 步）**
- 残差定义：`e_0 = z_clean - x_cond`
- 前向过程：$e_t = (1 - \eta_t) \cdot e_0 + \kappa \cdot \sqrt{\eta_t} \cdot \varepsilon$
- 后验分布：$\mu = \frac{\eta_{t-1}}{\eta_t} e_t + \frac{\alpha_t}{\eta_t} e_0$，
  $\sigma^2 = \kappa^2 \frac{\eta_{t-1}}{\eta_t} \alpha_t$
- 模型输入：`e_t + x_cond`（即噪声残差加回结构骨架）
- 模型输出：预测的干净残差 `ê_0`
- 最终重建：`z_clean = x_cond + e_0`

### 2.3 与原始 DiT 的兼容性

| 参数           | 原始 DiT          | 本项目 ResShift      |
|--------------|------------------|--------------------|
| `in_channels`| 4                | 4（不变）             |
| `learn_sigma`| True             | True（不变）          |
| `out_channels`| 8               | 8（取前4通道作为残差预测）  |
| `num_classes` | 1000            | 1000（不变）          |
| `x_cond`     | 不使用             | 通过 token 拼接，复用 x_embedder |
| 新增参数       | —               | 无（完全复用原始权重）      |

`x_cond` 通过与 `x` 共享的 `x_embedder` 嵌入后在 token 维度拼接，
Transformer 处理后只取前半 token 输出。**不引入任何新参数**。

---

## 3. 文件结构

```
DiT/
├── diffusion/
│   ├── __init__.py              # create_resshift_diffusion() 工厂函数
│   └── gaussian_diffusion.py    # ResShiftDiffusion 类 + schedule 函数
├── models.py                    # DiT 模型（支持 x_cond 拼接）
├── train_resshift.py            # 训练脚本（DDP，VAE + DDPM→SVD→ResShift）
├── sample_resshift.py           # 推理采样脚本（DDPM partial→SVD→ResShift→VAE decode）
├── test_resshift.py             # 端到端自动测试脚本
├── signal.txt                   # 任务控制信号文件
├── task.txt                     # 任务列表文件
└── README_resshift.md           # 本文档
```

---

## 4. 确定可运行的指令

### 4.1 运行自动测试（验证代码正确性）

```bash
cd /home/user/DiT && python test_resshift.py
```

此命令会依次测试：
1. ResShift schedule 生成
2. DDPM q_sample 在 0.5T + SVD 退化管线
3. ResShift 前向加噪过程
4. 完整训练循环（DDPM→SVD→ResShift，5步合成数据）
5. 完整推理采样（15步 ResShift 反向去噪）
6. 模型保存/加载一致性

预期输出：`ALL TESTS PASSED!`

### 4.2 使用真实数据训练（需要 ImageNet 格式数据集 + VAE）

权重加载使用 `download.py` 中的 `find_model()` 函数，与原始 DiT 的 `train.py` / `sample.py` 保持一致。
支持以下方式：
- 官方预训练模型名（如 `"DiT-XL-2-256x256.pt"`），会自动下载
- 本地 `train.py` 保存的 checkpoint 路径（自动提取 `"ema"` 键）
- 直接的 state_dict 文件路径

```bash
# 单卡训练（基于 DiT 预训练权重微调）
cd /home/user/DiT
torchrun --nnodes=1 --nproc_per_node=1 train_resshift.py \
    --data-path /path/to/imagenet/train \
    --model DiT-XL/2 \
    --image-size 256 \
    --global-batch-size 32 \
    --n-timestep 15 \
    --kappa 1.0 \
    --svd-ratio 0.25 \
    --vae ema \
    --pretrained-ckpt DiT-XL-2-256x256.pt \
    --epochs 100

# 多卡训练（如 4 卡）
torchrun --nnodes=1 --nproc_per_node=4 train_resshift.py \
    --data-path /path/to/imagenet/train \
    --model DiT-XL/2 \
    --image-size 256 \
    --global-batch-size 128 \
    --n-timestep 15 \
    --kappa 1.0 \
    --svd-ratio 0.25 \
    --pretrained-ckpt DiT-XL-2-256x256.pt
```

### 4.3 使用训练好的模型推理

推理脚本同样使用 `find_model()` 加载权重。如果 checkpoint 中包含 `"args"` 键（由 `train_resshift.py` 保存），
则自动使用训练时的参数；否则使用命令行参数作为兜底配置。

```bash
cd /home/user/DiT

# 从 train_resshift.py 保存的 checkpoint 推理（自动读取训练参数）
python sample_resshift.py \
    --ckpt results_resshift/000-DiT-XL-2-resshift/checkpoints/0010000.pt \
    --class-labels 207,360,387,974,88,979,417,279 \
    --cfg-scale 4.0 \
    --ddpm-steps 250 \
    --vae mse \
    --output-dir ./output_resshift

# 从原始预训练权重推理（需手动指定模型配置）
python sample_resshift.py \
    --ckpt DiT-XL-2-256x256.pt \
    --model DiT-XL/2 \
    --image-size 256 \
    --n-timestep 15 \
    --kappa 1.0 \
    --svd-ratio 0.25 \
    --class-labels 207,360,387,974 \
    --cfg-scale 4.0 \
    --vae mse
```

---

## 5. 关键参数说明

| 参数               | 默认值           | 说明                                                    |
|-------------------|-----------------|-------------------------------------------------------|
| `--n-timestep`    | 15              | ResShift 扩散步数（远少于 DDPM 的 1000 步）                  |
| `--kappa`         | 1.0             | 噪声缩放系数，控制 ResShift 前向过程中噪声强度                   |
| `--svd-ratio`     | 0.25            | SVD 保留的奇异值比例（25%）                                  |
| `--model`         | DiT-XL/2        | DiT 模型规格                                             |
| `--image-size`    | 256             | 输入图像分辨率                                             |
| `--vae`           | ema             | VAE 权重版本（ema/mse）                                   |
| `--pretrained-ckpt`| None           | 预训练 DiT 权重路径，用于微调                                  |
| `--cfg-scale`     | 4.0             | DDPM 阶段的 classifier-free guidance 强度                 |
| `--ddpm-steps`    | 250             | DDPM 采样总步数（实际运行 50% = 125 步）                      |

---

## 6. 测试验证结果

以下是 `python test_resshift.py` 的实际运行结果：

```
============================================================
  ResShift Diffusion + DiT — End-to-End Test Suite
  Pipeline: DDPM → SVD → ResShift (VAE latent space)
============================================================

[1/6] Testing ResShift schedule...
  sqrt_etas range: [0.010000, 0.990000]
  etas range: [0.000100, 0.980100]
  PASSED

[2/6] Testing DDPM q_sample at 0.5T + SVD degradation...
  DDPM q_sample at t=500: z_noisy std = 1.0016
  SVD 25%: mean diff from z_noisy = 0.4875 (should be > 0)
  PASSED

[3/6] Testing ResShift forward process (q_sample)...
  t=0 (no noise): max diff from expected = 2.38e-07 -- OK
  t=T-1: e_t std = 0.9849
  PASSED

[4/6] Testing training loop (DDPM→SVD→ResShift, 5 steps)...
  Step 1: loss = 1.285865
  Step 2: loss = 1.294991
  Step 3: loss = 1.255701
  Step 4: loss = 1.272956
  Step 5: loss = 1.297593
  PASSED

[5/6] Testing sampling loop (full inference pipeline)...
ResShift sampling: 100%|████████████████████████| 15/15 [00:00<00:00, 116.80it/s]
  Output shape: torch.Size([2, 4, 32, 32])
  Output range: [-3.6611, 2.7189]
  Mean diff from x_cond: 0.017774
  PASSED

[6/6] Testing checkpoint save/load...
  Max diff between original and loaded model: 0.00e+00
  PASSED

============================================================
  ALL TESTS PASSED!
============================================================
```
