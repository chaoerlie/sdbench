from evaluate import *
import ImageReward as RM
import torch
import clip
import os
import json
from tqdm import tqdm

def get_folders_and_prompts(folder_path, prompt_file):
    folder_paths = []
    
    # 读取提示词
    with open(prompt_file, 'r') as f:
        prompts = f.readlines()
    
    # 获取文件夹路径
    for folder in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, folder)
        if os.path.isdir(folder_path_full):
            folder_paths.append((folder, folder_path_full))  # 保存文件夹名称和路径
    
    return folder_paths, prompts

def load_existing_results(output_json):
    # 如果 JSON 文件已经存在，加载已保存的结果
    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            return json.load(f)
    else:
        return {}

def compute_and_save_scores(output_dir, prompt_file, output_json):
    # 加载现有结果
    results = load_existing_results(output_json)
    folder_paths, prompts = get_folders_and_prompts(output_dir, prompt_file)
    
    # 加载 ImageReward 模型
    model = RM.load("ImageReward-v1.0")

    # 遍历每个文件夹，计算分数
    print("Processing folders and images...")
    for folder_name, folder_path in tqdm(folder_paths, desc="Processing folders"):
        # 如果当前文件夹已经处理过，跳过
        if folder_name in results:
            print(f"Skipping folder {folder_name}, already processed.")
            continue
        
        folder_results = []
        print(f"Processing folder: {folder_path}")
        
        # 获取该文件夹中的所有图像文件
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        
        # 根据文件夹名称最后的数字来选择对应的提示词
        try:
            prompt_index = int(folder_name.split('_')[-1]) - 1  # 假设文件夹名的最后部分是数字
            prompt = prompts[prompt_index].strip()
        except (ValueError, IndexError):
            print(f"Error: Unable to determine prompt for folder '{folder_name}'. Skipping this folder.")
            continue  # 如果有错误，跳过该文件夹
        
        # 遍历文件夹中的每个图像并计算相应的分数
        for image_file in tqdm(image_files, desc=f"Processing images in {folder_name}", leave=False):
            image_path = os.path.join(folder_path, image_file)
            
            # 计算 HPS
            hps = calculate_hps(image_path, prompt, hpc_path="/home/ps/sdbench/models/hps/hpc.pt")
            
            # 计算 ImageReward
            rewards = model.score(prompt, image_path)
            
            # 计算 ClipScore
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_clip, transform = clip.load("ViT-B/32", device=device, jit=False)
            model_clip.eval()
            clip_score = get_clip_score_for_image_and_prompt(image_path, prompt, model_clip, device)
            
            # 保存当前图像的计算结果
            folder_results.append({
                'image_name': image_file,
                'prompt': prompt,
                'hps': hps,
                'ImageReward': rewards,
                'ClipScore': clip_score
            })
        
        # 保存当前文件夹的结果到结果字典中
        results[folder_name] = folder_results
        
        # 每处理完一个文件夹后保存结果到 JSON 文件
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results for {folder_name} saved.")

    print(f"All results saved to {output_json}")

if __name__ == "__main__":
    output_json = "/home/ps/sdbench/outputs/results.json"
    output_dir = "/home/ps/sdbench/outputs"
    prompt_file = "/home/ps/sdbench/outputs/prompt.txt"
    compute_and_save_scores(output_dir, prompt_file, output_json)