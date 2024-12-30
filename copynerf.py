import os
import random
import shutil

# 源文件夹路径
source_folder = "C:/Users/65309/Desktop/nerfCambridge/GreatCourt"
# 目标文件夹路径1，用于存放3/10的文件
destination_folder1 = "C:/Users/65309/Desktop/nerfCambridge/query/GreatCourt"
# 目标文件夹路径2，用于存放7/10的文件
destination_folder2 = "C:/Users/65309/Desktop/nerfCambridge/database/GreatCourt"


image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg') or f.endswith('.png')]
# 计算要选择的文件数量
total_files = len(image_files)
num_files_to_copy1 = total_files * 3 // 10
num_files_to_copy2 = total_files - num_files_to_copy1

# 创建目标文件夹
os.makedirs(destination_folder1, exist_ok=True)
os.makedirs(destination_folder2, exist_ok=True)

# 获取所有文件列表
files_list = list(range(total_files))

# 随机选择要复制的文件
files_to_copy1 = random.sample(files_list, num_files_to_copy1)
files_to_copy2 = [file_index for file_index in files_list if file_index not in files_to_copy1]

# 复制3/10的文件
for file_index in files_to_copy1:
    image_name = f"{file_index:05d}.png"
    txt_name = f"pose_{file_index:05d}.txt"
    source_path = os.path.join(source_folder, image_name)
    dest_path = os.path.join(destination_folder1, image_name)
    shutil.copyfile(source_path, dest_path)
    source_path = os.path.join(source_folder, txt_name)
    dest_path = os.path.join(destination_folder1, txt_name)
    shutil.copyfile(source_path, dest_path)

print(f"{num_files_to_copy1} files copied to the first destination folder.")

# 复制剩余的7/10的文件
for file_index in files_to_copy2:
    image_name = f"{file_index:05d}.png"
    txt_name = f"pose_{file_index:05d}.txt"
    source_path = os.path.join(source_folder, image_name)
    dest_path = os.path.join(destination_folder2, image_name)
    shutil.copyfile(source_path, dest_path)
    source_path = os.path.join(source_folder, txt_name)
    dest_path = os.path.join(destination_folder2, txt_name)
    shutil.copyfile(source_path, dest_path)

print(f"{num_files_to_copy2} files copied to the second destination folder.")
