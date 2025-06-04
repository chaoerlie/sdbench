#基于evaluate_hps.py修改
import torch
import clip
from PIL import Image


def calculate_hps(image_path, prompt, hpc_path, device="cuda"):
    """
    计算 HPS (Hierarchical Precision Score)
    :param image_paths: image file paths
    :param prompt: Text prompt for comparison (e.g., "your prompt here")
    :param hpc_path: Path to the HPC model checkpoint file (e.g., "path/to/hpc.pth")
    :param device: Device to run on ("cuda" or "cpu")
    :return: HPS scores (torch.Tensor)
    """
    # 检查设备
    device = device if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    model, preprocess = clip.load("ViT-L/14", device=device)
    params = torch.load(hpc_path)['state_dict']
    model.load_state_dict(params)
    model = model.to(device).eval()
    
    # 预处理图像
    images = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # 处理文本
    text = clip.tokenize([prompt]).to(device)
    
    # 计算特征和 HPS
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)

        # 特征归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 计算 HPS
        hps = image_features @ text_features.T

    return hps.item()
