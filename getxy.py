#从pose中提取xy保存
import os
import numpy as np

# 原始文件夹和新文件夹路径
original_folder = 'C:/Users/65309/Desktop/Cambridge_nerf/poses4'
new_folder = 'C:/Users/65309/Desktop/Cambridge_nerf/poses'

# 创建新文件夹（如果不存在）
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# 遍历原始文件夹中的所有txt文件
for filename in os.listdir(original_folder):
    if filename.endswith('.txt'):
        original_filepath = os.path.join(original_folder, filename)
        new_filepath = os.path.join(new_folder, filename)
        parts = filename.split("_")
        # 获取第一个部分作为目标字符部分
        target_string = parts[0]
        print(target_string)
        # 读取原始txt文件的内容并提取第一行第四列和第二行第四列的数据
        with open(original_filepath, 'r') as original_file:
            lines = original_file.readlines()
            if(target_string == "GreatCourt"):
                element_1 = float(lines[0].split()[3])
                element_2 = float(lines[1].split()[3])
            if(target_string == "KingsCollege"):
                element_1 = float(lines[0].split()[3])+1000
                element_2 = float(lines[1].split()[3])+1000
            if (target_string == "OldHospital"):
                element_1 = float(lines[0].split()[3])+2000
                element_2 = float(lines[1].split()[3])+2000
            if(target_string == "ShopFacade"):
                element_1 = float(lines[0].split()[3])+3000
                element_2 = float(lines[1].split()[3])+3000
            if(target_string == "StMarysChurch"):
                element_1 = float(lines[0].split()[3])+4000
                element_2 = float(lines[1].split()[3])+4000
            pose_loc = np.array([[element_1, element_2]])

        # 将提取的数据保存到新文件夹中与原文件同名的txt文件中
        np.savetxt(new_filepath, pose_loc)
