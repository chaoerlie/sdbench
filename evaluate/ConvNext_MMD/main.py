import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import os

from . import embedding, distance, io_util
from .io_util import CMMDDataset


def perturb_state_dict(state_dict, scale_dict=None, default_scale=1e-4):
    perturbed_state_dict = {}
    for name, param in state_dict.items():
        if not torch.is_floating_point(param):
            perturbed_state_dict[name] = param
            continue
        scale = default_scale
        if scale_dict:
            for key, s in scale_dict.items():
                if key in name:
                    scale = s
                    break
        noise = torch.randn_like(param) * scale
        perturbed_state_dict[name] = param + noise
    return perturbed_state_dict


def generate_perturbed_models_conv(base_model, num_perturbs=5, scale_dict=None, perturb_scale=1e-3):
    """
    根据 base_model 生成多个扰动模型副本。
    """
    perturbed_models = []
    base_model = base_model._model
    model_config = base_model.config
    base_state_dict = deepcopy(base_model.state_dict())
    for _ in tqdm(range(num_perturbs), desc="🎨 生成扰动模型"):
        perturbed_dict = perturb_state_dict(base_state_dict, scale_dict, perturb_scale)
        perturbed_model = base_model.__class__(model_config).eval()
        perturbed_model.load_state_dict(perturbed_dict, strict=False)
        if torch.cuda.is_available():
            perturbed_model = perturbed_model.cuda()
        perturbed_models.append(perturbed_model)
    return perturbed_models


def load_and_preprocess_image(img_path, input_size):
    """
    从路径加载图像，并缩放到模型所需尺寸，返回 shape=(1, H, W, 3)
    """
    dataset = CMMDDataset(img_path, reshape_to=input_size, is_single_file=True)
    image_np = dataset[0] / 255.0
    return np.expand_dims(image_np, axis=0)


def compute_cmmd_single_image_with_models(
    embedding_model,
    perturbed_models,
    single_img_path,
    ref_embs
):
    """
    使用多个扰动模型对单张图片进行特征提取，并与参考集计算 MMD 距离。

    Args:
        embedding_model: ConvNeXtV2EmbeddingModel 实例。
        perturbed_models: 由 generate_perturbed_models_conv 返回的扰动模型列表。
        single_img_path: 单张图像路径。
        ref_embs: numpy.ndarray，参考图像嵌入。

    Returns:
        float: MMD 距离值。
    """
    # ✅ 加载并预处理图像
    image_np = load_and_preprocess_image(single_img_path, embedding_model.input_image_size)

    # ✅ 遍历扰动模型并提取该图的多组特征
    single_embs = []
    for model in tqdm(perturbed_models, desc="📦 从convnext模型提取嵌入"):
        embedding_model._model = model
        with torch.no_grad():
            emb = embedding_model.embed(image_np).cpu().numpy()[0]
        single_embs.append(emb)

    single_embs = np.stack(single_embs, axis=0)

    # ✅ 对齐参考集大小
    M = single_embs.shape[0]
    if ref_embs.shape[0] >= M:
        idx = np.random.choice(ref_embs.shape[0], size=M, replace=False)
        ref_embs_sampled = ref_embs[idx]
    else:
        N = ref_embs.shape[0]
        idx = np.random.choice(M, size=N, replace=False)
        single_embs = single_embs[idx]
        ref_embs_sampled = ref_embs

    return distance.mmd(ref_embs_sampled, single_embs).item()


def collect_single_image_from_each_subfolder(parent_dir):
    """
    从每个子文件夹中选取第一张图片，返回全路径列表。
    """
    valid_ext = (".png", ".jpg", ".jpeg")
    img_paths = []

    for subfolder in sorted(os.listdir(parent_dir)):
        subfolder_path = os.path.join(parent_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for file in sorted(os.listdir(subfolder_path)):
            if file.lower().endswith(valid_ext):
                img_paths.append(os.path.join(subfolder_path, file))
                break
    return img_paths
