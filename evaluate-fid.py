import os
import json
from tqdm import tqdm
from evaluate import *


def get_folders(folder_path):
    """获取指定文件夹下的所有子文件夹"""
    folder_paths = []
    for folder in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, folder)
        if os.path.isdir(folder_path_full):
            folder_paths.append(folder_path_full)
    return folder_paths

def compute_and_save_scores(output_dir, result_fid_json):
    folder_paths = get_folders(output_dir)
    
    results = {}

    # 遍历每个文件夹，计算 CMMD 和 FID 分数
    print("Processing folders...")
    for folder_path in tqdm(folder_paths, desc="Processing folders"):
        folder_name = os.path.basename(folder_path)
        print(f"Processing folder: {folder_name}")
        
        # 计算 CMMD 和 FID 分数（针对整个文件夹）
        CMMD_score = calc_CMMD_Score('result/true', folder_path)
        fid_score = calc_Fid('result/true', folder_path)
        
        # 将分数添加到结果字典中
        results[folder_name] = {
            'CMMD_score': CMMD_score,
            'FID_score': fid_score
        }

    # 保存结果到 JSON 文件
    with open(result_fid_json, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_fid_json}")

if __name__ == "__main__":
    output_dir = "/home/ps/sdbench/benc/all"  # 目标文件夹路径
    result_fid_json = "/home/ps/sdbench/result_fid.json"  # 保存结果的 JSON 文件路径
    compute_and_save_scores(output_dir, result_fid_json)
