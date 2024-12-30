import numpy as np
import torch

def pose_tensor_to_pose_representations(pose_tensor):
    # 获取张量的形状
    num_poses, _, _ = pose_tensor.shape

    # 初始化结果数组
    pose_representations_euler = np.zeros((num_poses, 6))

    for i in range(num_poses):
        # 提取位置信息
        position = pose_tensor[i, :3, 3]

        # 提取旋转信息
        rotation_matrix = pose_tensor[i, :3, :3]

        # 欧拉角表示
        euler_angles = np.array([0, 0, 0])
        euler_angles = np.degrees(np.around(np.array([
            np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]),
            np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)),
            np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        ]), decimals=6))
        pose_representations_euler[i] = np.concatenate((position, euler_angles))

    return pose_representations_euler

# 示例用法
pose_matrices = np.random.rand(10, 4, 4)  # 生成随机的位姿矩阵数组，假设有10个位姿
euler_poses= pose_tensor_to_pose_representations(pose_matrices)

print("Euler Pose Representations:")
print(euler_poses.shape)


