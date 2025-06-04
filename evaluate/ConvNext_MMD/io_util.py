"""IO utilities."""

import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import tqdm
import os
import sys

Image.MAX_IMAGE_PIXELS = None  # 关闭限制

class CMMDDataset(Dataset):
    def __init__(self, path, reshape_to, max_count=-1, is_single_file=False):
        self.path = path
        self.reshape_to = reshape_to
        self.max_count = max_count
        self.is_single_file = is_single_file
        
        if is_single_file:
            self.img_path_list = [path]
        else:
            img_path_list = self._get_image_list()
            if max_count > 0:
                img_path_list = img_path_list[:max_count]
            self.img_path_list = img_path_list

    def __len__(self):
        return len(self.img_path_list)

    def _get_image_list(self):
        ext_list = ["png", "jpg", "jpeg"]
        image_list = []
        for ext in ext_list:
            image_list.extend(glob.glob(f"{self.path}/*.{ext}"))
            image_list.extend(glob.glob(f"{self.path}/*.{ext.upper()}"))
        image_list.sort()
        return image_list

    def _center_crop_and_resize(self, im, size):
        w, h = im.size
        l = min(w, h)
        top = (h - l) // 2
        left = (w - l) // 2
        box = (left, top, left + l, top + l)
        im = im.crop(box)
        return im.resize((size, size), resample=Image.BICUBIC)

    def _read_image(self, path, size):
        im = Image.open(path).convert("RGB")
        if size > 0:
            im = self._center_crop_and_resize(im, size)
        return np.asarray(im).astype(np.float32)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        x = self._read_image(img_path, self.reshape_to)
        if x.ndim == 3:
            return x
        elif x.ndim == 2:
            return np.tile(x[..., np.newaxis], (1, 1, 3))

def compute_embeddings_for_dir(
    img_dir,
    embedding_model,
    batch_size,
    max_count=-1,
):
    dataset = CMMDDataset(img_dir, reshape_to=embedding_model.input_image_size, max_count=max_count)
    count = len(dataset)
    print(f"计算目录 {img_dir} 中 {count} 张图片的embeddings.")

    dataloader = DataLoader(dataset, batch_size=batch_size)

    all_embs = []
    with tqdm.tqdm(total=count, desc="📦 提取嵌入", unit="img") as pbar:
        for batch in dataloader:
            image_batch = batch.numpy()
            image_batch = image_batch / 255.0

            if np.min(image_batch) < 0 or np.max(image_batch) > 1:
                raise ValueError(
                    f"图片值应该在[0, 1]范围内，但发现值在: [{np.min(image_batch)}, {np.max(image_batch)}]"
                )

            embs = np.asarray(embedding_model.embed(image_batch))
            all_embs.append(embs)

            pbar.update(len(image_batch))  # 更新已处理的图像数

    all_embs = np.concatenate(all_embs, axis=0)
    return all_embs

def compute_embedding_for_single_image(
    img_path,
    embedding_model,
):
    dataset = CMMDDataset(
        img_path, 
        reshape_to=embedding_model.input_image_size, 
        is_single_file=True
    )
    
    image = dataset[0]
    image = image / 255.0

    if np.min(image) < 0 or np.max(image) > 1:
        raise ValueError(
            f"图片值应该在[0, 1]范围内，但发现值在: [{np.min(image)}, {np.max(image)}]"
        )
    
    image = np.expand_dims(image, 0)
    emb = embedding_model.embed(image)
    
    return np.asarray(emb)