import json
import os
import numpy as np


def save_transform_matrices(json_path):
    # 加载 JSON 文件
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 遍历每个 frame
    for frame in data['frames']:
        file_path = frame['file_path']
        transform_matrix = np.array(frame['transform_matrix'])

        # 获取图片文件名，例如 frame_00594
        file_name = os.path.basename(file_path).replace('.png', '')

        # 创建保存路径，例如 poses4/frame_00594.txt
        save_path = os.path.join('F:/waymo/scan15/poses4', f"{file_name}.txt")

        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存矩阵到 .txt 文件
        np.savetxt(save_path, transform_matrix, fmt='%.8f')

    print("Transform matrices saved successfully.")


# 使用时替换为你的 JSON 文件路径
json_file_path = 'F:/waymo/scan15/transforms.json'
save_transform_matrices(json_file_path)
