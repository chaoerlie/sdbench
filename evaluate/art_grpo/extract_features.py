import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel, ConvNextV2Model, ConvNextV2ImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 加载模型 ===
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

convnext_model = ConvNextV2Model.from_pretrained("facebook/convnextv2-base-1k-224").to(device)
convnext_processor = ConvNextV2ImageProcessor.from_pretrained("facebook/convnextv2-base-1k-224")

# === 图像路径 ===
with open("data/image_paths.json", "r") as f:
    image_paths = json.load(f)  # dict: {id: path}

# === 提前确定特征 shape ===
clip_dim = clip_model.visual.projection.out_features
convnext_dim = convnext_model.config.hidden_sizes[-1]
N = len(image_paths)

clip_feats = np.zeros((N, clip_dim), dtype=np.float32)
conv_feats = np.zeros((N, convnext_dim), dtype=np.float32)

# === 图像预处理 + 特征提取 ===
for idx, (img_id, img_path) in enumerate(tqdm(image_paths.items())):
    try:
        image = Image.open(img_path).convert("RGB")

        # CLIP
        clip_inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            clip_feat = clip_model.get_image_features(**clip_inputs)
        clip_feats[idx] = clip_feat.cpu().numpy()

        # ConvNeXtV2
        conv_inputs = convnext_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            conv_feat = convnext_model(**conv_inputs).pooler_output  # [1, D]
        conv_feats[idx] = conv_feat.cpu().numpy()

    except Exception as e:
        print(f"Error at {img_id}: {e}")

# === 保存特征 ===
np.save("data/features_clip.npy", clip_feats)
np.save("data/features_convnext.npy", conv_feats)
print("✅ 特征保存完毕！")
