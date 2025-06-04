from transformers import AutoImageProcessor, ConvNextV2Model, AutoConfig
import torch
import numpy as np
import os

_CONVNEXT_MODEL_NAME = "facebook/convnextv2-base-1k-224"
_CUDA_AVAILABLE = torch.cuda.is_available()

def _resize_bicubic(images, size):
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bicubic")
    images = images.permute(0, 2, 3, 1).numpy()
    return images

class ConvNextV2EmbeddingModel:
    """ConvNeXtV2 image embedding calculator."""
    def __init__(self,checkpoint_path=None):
        """
        Args:
            checkpoint_path: optional path to your fine-tuned .pth model (from training with timm)
        """
        self.image_processor = AutoImageProcessor.from_pretrained(_CONVNEXT_MODEL_NAME)
        self.input_image_size = self.image_processor.size["shortest_edge"]

        if checkpoint_path is not None and checkpoint_path.endswith(".pth") and os.path.exists(checkpoint_path):
            print(f"加载自定义模型权重: {checkpoint_path}")
            self._model = self._load_finetuned_model(checkpoint_path)
        else:
            print("加载默认预训练模型: facebook/convnextv2-base-1k-224")
            self._model = ConvNextV2Model.from_pretrained(_CONVNEXT_MODEL_NAME).eval()

        # self._model = ConvNextV2Model.from_pretrained(_CONVNEXT_MODEL_NAME).eval()

        if _CUDA_AVAILABLE:
            self._model = self._model.cuda()

        # 输入尺寸，例如 224x224
        self.input_image_size = self.image_processor.size["shortest_edge"]

    def _load_finetuned_model(self, checkpoint_path):
        """Loads a fine-tuned ConvNeXtV2 model from a checkpoint."""
        config = AutoConfig.from_pretrained(_CONVNEXT_MODEL_NAME)
        model = ConvNextV2Model(config)
        # 加载 state_dict
        state_dict = torch.load(checkpoint_path, map_location="cuda" if _CUDA_AVAILABLE else "cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
        return model.eval()

    @torch.no_grad()
    def embed(self, images):
        """Computes ConvNeXtV2 embeddings for the given images."""
        images = _resize_bicubic(images, self.input_image_size)
        images = np.clip(images, 0.0, 1.0)
        inputs = self.image_processor(
            images=images,
            return_tensors="pt"
        )
        if _CUDA_AVAILABLE:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # 提取最后一层的池化特征
        outputs = self._model(**inputs)
        features = outputs.pooler_output.cpu()  # shape (B, hidden_dim)
        features /= torch.linalg.norm(features, axis=-1, keepdims=True)
        return features
