import torch
import torchvision
import os;

from pytorch_fid import fid_score

def count_images(folders):
    return len([f for f in os.listdir(folders) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

def calc_Fid(real_images_folder,generated_images_folder, batch_size=50, device=None, dims=2048):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_real = count_images(real_images_folder)
    num_generated = count_images(generated_images_folder)
    print("真实图像个数:",num_real)
    print("生成图像个数:",num_generated)

    fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],batch_size=batch_size, device=device, dims=dims)
    return fid_value

if __name__ == '__main__':
    real_images_folder = '/home/ps/sdbench/Shanshui/Shanshui'
    generated_images_folder = '/home/ps/sdbench/benc/test/1'
    fid_value = calc_Fid(real_images_folder,generated_images_folder, batch_size=50, device='cuda:0', dims=2048)
    print('FID value:', fid_value)