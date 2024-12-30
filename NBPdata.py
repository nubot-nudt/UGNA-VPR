#选择图片作为train的database和test的database   selected_indices = [i for i in range(len(image_files)) if (i + 1) % 4 == 0]

import os
import shutil

folders = ["scan01", "scan02", "scan03", "scan04", "scan05"]
"""
folders = ["europa", "lk2", "lwp", "rathaus", "schloss", "st", "stjacob", "stjohann"]
"""
for folder_name in folders:
    # 定义原始图片文件夹和文本文件夹的路径
    image_folder = "C:/Users/65309/Desktop/NEU_night_rename/"+folder_name+"/images"
    txt_folder = "C:/Users/65309/Desktop/NEU_night_rename/"+folder_name+"/poses4"

    # 定义新的图片文件夹和文本文件夹的路径
    new_image_folder_1 = "C:/Users/65309/Desktop/NBPval/"+folder_name+"/images"
    new_txt_folder_1 = "C:/Users/65309/Desktop/NBPval/"+folder_name+"/poses"
    new_image_folder_2 = "C:/Users/65309/Desktop/NBPtrain/"+folder_name+"/images"
    new_txt_folder_2 = "C:/Users/65309/Desktop/NBPtrain/"+folder_name+"/poses"

    os.makedirs(new_image_folder_1, exist_ok=True)
    os.makedirs(new_txt_folder_1, exist_ok=True)
    os.makedirs(new_image_folder_2, exist_ok=True)
    os.makedirs(new_txt_folder_2, exist_ok=True)
    # 获取图片文件夹中的所有文件名
    image_files = os.listdir(image_folder)

    # 获取txt文件夹中的所有文件名
    txt_files = os.listdir(txt_folder)

    # 确保文件名列表按照相同的顺序排序
    image_files.sort()
    txt_files.sort()

    # 每隔四个元素选取一个元素
    selected_indices = [i for i in range(len(image_files)) if (i + 1) % 3 == 0]

    # 将选定的文件复制到新的文件夹中
    for idx in selected_indices:
        image_file = image_files[idx]
        txt_file = txt_files[idx]

        # 构建原始文件的完整路径
        image_path = os.path.join(image_folder, image_file)
        txt_path = os.path.join(txt_folder, txt_file)

        # 复制图片文件
        shutil.copy(image_path, new_image_folder_1)

        # 复制文本文件
        shutil.copy(txt_path, new_txt_folder_1)

    # 处理剩余的文件
    for idx in range(len(image_files)):
        if idx not in selected_indices:
            image_file = image_files[idx]
            txt_file = txt_files[idx]

            # 构建原始文件的完整路径
            image_path = os.path.join(image_folder, image_file)
            txt_path = os.path.join(txt_folder, txt_file)

            # 复制图片文件
            shutil.copy(image_path, new_image_folder_2)

            # 复制文本文件
            shutil.copy(txt_path, new_txt_folder_2)