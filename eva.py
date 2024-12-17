import json
import numpy as np
import os

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_json(output_json):
    # 加载结果文件
    results = load_json(output_json)

    # 用于存储每种评估方法的统计数据
    method_scores = {}

    # 遍历每个文件夹，计算每种评估方法的统计信息
    for folder_name, folder_results in results.items():
        # 从文件夹名称提取训练方法
        method_name = folder_name.split('_')[0]
        
        if method_name not in method_scores:
            method_scores[method_name] = {'hps': [], 'ImageReward': [], 'ClipScore': []}

        # 统计当前文件夹中的每个评估方法
        for image_result in folder_results:
            method_scores[method_name]['hps'].append(image_result['hps'])
            method_scores[method_name]['ImageReward'].append(image_result['ImageReward'])
            method_scores[method_name]['ClipScore'].append(image_result['ClipScore'])

    # 输出每种评估方法在每个训练方法中的统计数据
    print("\n每种评估方法在每个训练方法中的统计数据:")
    for method_name, scores in method_scores.items():
        print(f"\n训练方法：{method_name}")
        
        # 对每个评分项计算均值、方差、最小值和最大值
        for score_type, score_list in scores.items():
            mean_score = np.mean(score_list)
            variance_score = np.var(score_list)
            min_score = np.min(score_list)
            max_score = np.max(score_list)

            print(f"  {score_type} 分数统计:")
            print(f"    平均分: {mean_score:.2f}")
            print(f"    方差: {variance_score:.2e}")
            print(f"    最低分: {min_score:.2f}")
            print(f"    最高分: {max_score:.2f}")

def analyze_json2(output_json):
    # 加载结果文件
    results = load_json(output_json)

    # 用于存储每个评估方法的分数数据
    score_data = {'hps': [], 'ImageReward': [], 'ClipScore': []}
    method_scores = {}

    # 遍历每个文件夹，计算每种评估方法的分数
    for folder_name, folder_results in results.items():
        # 从文件夹名称提取训练方法
        method_name = folder_name.split('_')[0]
        
        if method_name not in method_scores:
            method_scores[method_name] = {'hps': [], 'ImageReward': [], 'ClipScore': []}

        # 统计当前文件夹中的每个评估方法
        for image_result in folder_results:
            method_scores[method_name]['hps'].append(image_result['hps'])
            method_scores[method_name]['ImageReward'].append(image_result['ImageReward'])
            method_scores[method_name]['ClipScore'].append(image_result['ClipScore'])

            # 记录每个评估方法的分数
            score_data['hps'].append(image_result['hps'])
            score_data['ImageReward'].append(image_result['ImageReward'])
            score_data['ClipScore'].append(image_result['ClipScore'])

    # 输出每种评估方法在每个训练方法中的排名
    print("\n每种评估方法在每个训练方法中的排名:")
    for score_type in ['hps', 'ImageReward', 'ClipScore']:
        print(f"\n评估方法：{score_type}")

        # 为每个评估方法排名
        method_ranking = []
        for method_name, scores in method_scores.items():
            score_list = scores[score_type]
            
            # 计算该训练方法在该评估方法下的均值（作为该方法的得分）
            mean_score = np.mean(score_list)
            method_ranking.append((method_name, mean_score))

        # 根据均值对训练方法进行排序，降序
        method_ranking.sort(key=lambda x: x[1], reverse=True)

        # 输出排序结果
        for rank, (method_name, mean_score) in enumerate(method_ranking, start=1):
            print(f"  {rank}. {method_name}: {mean_score:.4f}")

    # 输出每种评估方法的总体排名
    print("\n汇总每种评估方法的总体排名:")
    for score_type in ['hps', 'ImageReward', 'ClipScore']:
        all_scores = score_data[score_type]
        
        # 为总体分数计算均值排名
        all_method_ranking = []
        for method_name, scores in method_scores.items():
            mean_score = np.mean(scores[score_type])
            all_method_ranking.append((method_name, mean_score))

        # 根据均值对训练方法进行排序，降序
        all_method_ranking.sort(key=lambda x: x[1], reverse=True)

        # 输出排序结果
        print(f"\n评估方法：{score_type}")
        for rank, (method_name, mean_score) in enumerate(all_method_ranking, start=1):
            print(f"  {rank}. {method_name}: {mean_score:.4f}")

if __name__ == "__main__":
    # 读取结果的 JSON 文件路径
    output_json = "/home/ps/sdbench/results_no1.json"
    
    # 分析 JSON 文件
    analyze_json(output_json)
    analyze_json2(output_json)
