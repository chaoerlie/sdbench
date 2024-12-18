from evaluate import *
import ImageReward as RM
import torch
import clip
import os
from pathlib import Path
import json
from tqdm import tqdm


def get_folders_and_prompts(folder_path, prompt_file):
    folder_paths = []
    
    # 读取提示词
    with open(prompt_file, 'r') as f:
        prompts = f.readlines()
    
    print(f"Looking in directory: {folder_path}")
    # 获取文件夹路径
    for folder in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, folder)
        print(f"Checking: {folder_path_full}")
        if os.path.isdir(folder_path_full):
            folder_paths.append((folder, folder_path_full))  # 保存文件夹名称和路径
    
    return folder_paths, prompts



def compute_and_save_scores(output_dir, prompt_file,output_json):
    folder_paths, prompts = get_folders_and_prompts(output_dir,prompt_file)
    
    # 加载 ImageReward 模型
    model = RM.load("ImageReward-v1.0")
    
    results = {}

    # 遍历每个文件夹，计算分数
    print("Processing folders and images...")
    for folder_path in tqdm(folder_paths, desc="Processing folders"):
        folder_name = os.path.basename(folder_path)
        folder_results = {}
        
        # 获取该文件夹中的所有图像文件
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        
        # 遍历文件夹中的每个图像和对应的提示词
        for idx, image_file in enumerate(tqdm(image_files, desc=f"Processing images in {folder_name}", leave=False)):
            # 使用 prompt_list 中的对应提示词
            if idx < len(prompts):
                prompt = prompts[idx].strip()
            else:
                continue  # 如果提示词数量少于图像数量，则跳过

            image_path = os.path.join(folder_path, image_file)
            
            # 计算 HPS
            hps = calculate_hps(image_path, prompt, hpc_path="/home/ps/sdbench/models/hps/hpc.pt")
            
            # 计算 ImageReward
            rewards = model.score(prompt, image_path)
            
            # 计算 PickScore
            # pick_score = calculate_PickScore(image_path, prompt)
            
            # 计算 ClipScore
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_clip, transform = clip.load("ViT-B/32", device=device, jit=False)
            model_clip.eval()
            clip_score = get_clip_score_for_image_and_prompt(image_path, prompt, model_clip, device)
            
            # 保存当前图像的计算结果
            folder_results[image_file] = {
                'prompt': prompt,
                'hps': hps,
                'ImageReward': rewards,
                'PickScore': pick_score,
                'ClipScore': clip_score
            }
        
        # 保存当前文件夹的结果
        results[folder_name] = folder_results

    # 将结果保存到JSON文件
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_json}")



if __name__ == "__main__":
    # #HPS
    image_path = "/home/ps/sdbench/outputs/reference/40.jpg"
    prompt = "traditional Chinese painting, misty mountains, calm river, small boat, pine trees, ink wash, soft green and gray tones, serene atmosphere"
    hpc_path = "/home/ps/sdbench/models/hps/hpc.pt"
    hps = calculate_hps(image_path, prompt, hpc_path)
    print("hps:",hps)

    #ImageReward
    model = RM.load("ImageReward-v1.0")
    rewards = model.score(prompt, image_path)
    print("ImageReward:", rewards)

    #PickScore
    pick_score = calculate_PickScore(image_path, prompt)
    print("PickScore:", pick_score)

    #ClipScore
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    clip_score = get_clip_score_for_image_and_prompt(image_path, prompt, model, device)
    print(f"CLIPScore for the image and prompt: {clip_score:.4f}")
    # output_json = "/home/ps/sdbench/outputs/results.json"
    # output_dir = "/home/ps/sdbench/outputs"
    # prompt_file = "/home/ps/sdbench/outputs/prompt.txt"
    # folder_paths, prompts = get_folders_and_prompts(output_dir,prompt_file)
    # print("folder_paths:",folder_paths)



