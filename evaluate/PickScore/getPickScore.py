from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

def calculate_PickScore(image_paths, prompt, device="cuda"):
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
    pil_images = [Image.open(path) for path in image_paths]
    image_inputs = processor(
        images=pil_images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)
    
    return probs.cpu().tolist()
