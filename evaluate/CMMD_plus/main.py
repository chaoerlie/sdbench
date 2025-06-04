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


def generate_perturbed_models_clip(base_model, num_perturbs=5, scale_dict=None, perturb_scale=1e-3):
    """
    根据 base_model 生成多个扰动模型副本（CLIP版本）
    """
    perturbed_models = []
    base_model = base_model._model
    model_config = base_model.config
    base_state_dict = deepcopy(base_model.state_dict())
    for _ in tqdm(range(num_perturbs), desc="🎨 生成扰动模型 (CLIP)"):
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
    使用多个扰动模型对单张图像提取嵌入 → 构成分布 → 与参考特征计算 MMD。
    """
    image_np = load_and_preprocess_image(single_img_path, embedding_model.input_image_size)

    single_embs = []
    for model in tqdm(perturbed_models, desc="📦 CLIP 提取扰动嵌入"):
        embedding_model._model = model
        with torch.no_grad():
            emb = embedding_model.embed(image_np).cpu().numpy()[0]
        single_embs.append(emb)

    single_embs = np.stack(single_embs, axis=0)

    M = single_embs.shape[0]
    if ref_embs.shape[0] >= M:
        ref_embs_sampled = ref_embs[np.random.choice(ref_embs.shape[0], size=M, replace=False)]
    else:
        idx = np.random.choice(M, size=ref_embs.shape[0], replace=False)
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

# import torch
# import numpy as np
# from tqdm import tqdm
# from copy import deepcopy
# import os
# from . import embedding, distance, io_util

# def perturb_state_dict(state_dict, scale_dict=None, perturb_scale=1e-4):
#     perturbed_state_dict = {}
#     for name, param in state_dict.items():
#         if not torch.is_floating_point(param):
#             perturbed_state_dict[name] = param
#             continue
#         scale = scale_dict.get(name, perturb_scale) if scale_dict else perturb_scale
#         noise = torch.randn_like(param) * scale
#         perturbed_state_dict[name] = param + noise
#     return perturbed_state_dict

# def generate_perturbed_models_clip(embedding_model, num_perturbs, scale_dict=None, perturb_scale=1e-4):
#     base_model = embedding_model._model
#     model_config = base_model.config
#     perturbed_models = []
#     for _ in tqdm(range(num_perturbs), desc="🔧 生成扰动模型"):
#         perturbed_dict = perturb_state_dict(deepcopy(base_model.state_dict()), scale_dict, perturb_scale)
#         perturbed_model = base_model.__class__(model_config).eval()
#         perturbed_model.load_state_dict(perturbed_dict, strict=False)
#         if torch.cuda.is_available():
#             perturbed_model = perturbed_model.cuda()
#         perturbed_models.append(perturbed_model)
#     return perturbed_models

# def sma_clipmmd(embedding_model, perturbed_models, img_list, ref_embs):
#     """
#     使用一批预先生成的扰动模型对图像列表做嵌入提取 + SMA 平均，再与 ref_embs 计算 MMD。
#     """
#     num_imgs = len(img_list)
#     sma_emb_list = [np.zeros_like(ref_embs[0]) for _ in range(num_imgs)]

#     for model in tqdm(perturbed_models, desc="🎨 SMA 聚合中"):
#         embedding_model._model = model
#         for i, img_path in enumerate(img_list):
#             emb = io_util.compute_embedding_for_single_image(img_path, embedding_model)[0]
#             sma_emb_list[i] += emb

#     for i in range(num_imgs):
#         sma_emb_list[i] /= len(perturbed_models)

#     final_embs = np.stack(sma_emb_list, axis=0)

#     # 匹配参考集大小
#     M = final_embs.shape[0]
#     if ref_embs.shape[0] >= M:
#         idx = np.random.choice(ref_embs.shape[0], size=M, replace=False)
#         ref_sample = ref_embs[idx]
#     else:
#         idx = np.random.choice(M, size=ref_embs.shape[0], replace=False)
#         final_embs = final_embs[idx]
#         ref_sample = ref_embs

#     return distance.mmd(ref_sample, final_embs).item()

# def collect_single_image_from_each_subfolder(image_dir):
#     """
#     遍历parent_dir下的每个子文件夹，从每个文件夹中找出第一张图片（png/jpg/jpeg），返回其完整路径。
    
#     返回：
#         List[str] -> 例如 ['/path/to/flux_1/xx.png', '/path/to/flux_2/xx.png']
#     """
#     valid_ext = (".png", ".jpg", ".jpeg")
#     img_paths = []

#     for subfolder in sorted(os.listdir(image_dir)):
#         subfolder_path = os.path.join(image_dir, subfolder)
#         if not os.path.isdir(subfolder_path):
#             continue

#         # 找该子文件夹下第一张图片
#         for file in sorted(os.listdir(subfolder_path)):
#             if file.lower().endswith(valid_ext):
#                 img_paths.append(os.path.join(subfolder_path, file))
#                 break  # 每个子文件夹只取1张
#     return img_paths


