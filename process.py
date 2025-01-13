
import os
import shutil

def extract_images(source_folder, target_folder, image_extensions=None):
    """
    从指定文件夹下的子文件夹中提取所有图片文件，并复制到新文件夹中

    :param source_folder: 源文件夹路径
    :param target_folder: 目标文件夹路径
    :param image_extensions: 图片文件的扩展名列表，默认为常见的图片类型
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

    # 创建目标文件夹（如果不存在的话）
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 检查文件扩展名是否是图片格式
            if any(file.lower().endswith(ext) for ext in image_extensions):
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(target_folder, file)

                # 如果目标文件夹已经有相同文件名的文件，重命名以避免覆盖
                if os.path.exists(target_file_path):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(target_file_path):
                        target_file_path = os.path.join(target_folder, f"{base}_{counter}{ext}")
                        counter += 1
                
                # 复制图片文件到目标文件夹
                shutil.copy(source_file_path, target_file_path)
                print(f"复制文件: {source_file_path} 到 {target_file_path}")

# 使用示例
source_folder = 'benc/sdxl'  # 修改为实际的源文件夹路径
target_folder = 'benc/all/sdxl'  # 修改为实际的目标文件夹路径
extract_images(source_folder, target_folder)
