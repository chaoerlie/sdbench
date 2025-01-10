import torch
import torchvision

from pytorch_fid import fid_score
# # 准备真实数据分布和生成模型的图像数据
# real_images_folder = 'result/sdlora'
# generated_images_folder = 'result/fluxlora'
# # 加载预训练的Inception-v3模型
# # inception_model = torchvision.models.inception_v3()
# # 定义图像变换

# # 计算FID距离值
# fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],batch_size=50, device='cuda:0', dims=2048, num_workers=1)

# print('FID value:', fid_value)


def calc_Fid(real_images_folder,generated_images_folder):
    fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],batch_size=50, device='cuda:0', dims=2048, num_workers=1)
    return fid_value