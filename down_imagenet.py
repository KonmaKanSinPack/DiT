# from datasets import load_dataset
# # 这会下载整个数据集到你的 ~/.cache/huggingface 目录
# dataset = load_dataset("imagenet-1k")

from datasets import load_dataset
import os
from tqdm import tqdm

print("正在从本地缓存加载 ImageNet...")
# 这里会自动读取你之前下载好的 .arrow 文件，不需要重新联网下载
ds_train = load_dataset("imagenet-1k", split="train")

save_dir = "./imagenet100/train"
os.makedirs(save_dir, exist_ok=True)

print("开始提取前 100 个类别构建 ImageNet-100...")

saved_count = 0
# 遍历数据集，只提取标签为 0 到 99 的图片
for item in tqdm(ds_train, desc="解压进度"):
    label = item['label']
    
    if label < 100:  # 核心拦截逻辑：只要前 100 个类
        class_dir = os.path.join(save_dir, str(label))
        # exist_ok=True 保证了即使文件夹存在也不会报错
        os.makedirs(class_dir, exist_ok=True)
        
        img = item['image']
        # 确保图片是 RGB 格式，防止 DiT 训练时通道数报错
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # 存为 JPEG
        img_path = os.path.join(class_dir, f"{saved_count}.jpg")
        img.save(img_path)
        saved_count += 1

print(f"\n大功告成！一共提取了 {saved_count} 张图片。")
print(f"你的训练数据已经完美准备在：{os.path.abspath(save_dir)}")