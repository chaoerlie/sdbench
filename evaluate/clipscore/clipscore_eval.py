import clip
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import sklearn.preprocessing
import warnings
from packaging import version


def preprocess_image(image_path, device, n_px=224):
    """
    预处理图片并转为 Tensor。
    """
    image = Image.open(image_path)
    transform = Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    image = transform(image).unsqueeze(0).to(device)  # 添加批次维度
    return image


def preprocess_text(prompt, model, device):
    """
    预处理文本 prompt，并将其转换为文本特征。
    """
    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text).cpu().numpy()
    return text_features


def get_clip_score_for_image_and_prompt(image_path, prompt, model, device="cuda"):
    """
    给定图片路径和文本提示，计算其 CLIPScore。
    """
    # 加载并预处理图片
    image_tensor = preprocess_image(image_path, device)
    
    # 提取图片特征
    with torch.no_grad():
        image_features = model.encode_image(image_tensor).cpu().numpy()
    
    # 预处理文本
    text_features = preprocess_text(prompt, model, device)
    
    # 归一化图片和文本特征
    if version.parse(np.__version__) < version.parse('1.21'):
        image_features = sklearn.preprocessing.normalize(image_features, axis=1)
        text_features = sklearn.preprocessing.normalize(text_features, axis=1)
    else:
        warnings.warn(
            'Due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'To exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        image_features = image_features / np.sqrt(np.sum(image_features**2, axis=1, keepdims=True))
        text_features = text_features / np.sqrt(np.sum(text_features**2, axis=1, keepdims=True))

    # 计算 CLIPScore（点积）
    clip_score = np.dot(image_features, text_features.T).item()  # 转为标量值

    return clip_score


def main():
    # 初始化 CLIP 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    # 输入图片路径和提示词
    image_path = "/home/ps/sdbench/outputs/fluxlora_30_1/20241211_164222_1.png"  # 替换为你的图片路径
    prompt = "chinese_painting, lotus flowers, with soft brushstrokes showing the flowers floating on a calm pond, surrounded by green leaves and some mist."  # 替换为你想要的提示词

    # 计算 CLIPScore
    clip_score = get_clip_score_for_image_and_prompt(image_path, prompt, model, device)

    print(f"CLIPScore for the image and prompt: {clip_score:.4f}")


if __name__ == '__main__':
    main()
