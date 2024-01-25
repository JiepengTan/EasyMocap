import numpy as np
from transforms3d.axangles import axangle2mat

def process_smpl_data(data):
    # 全局旋转和平移
    Rh = np.array(data['Rh'])
    Th = np.array(data['Th'])
    
    # 姿势参数
    poses = np.array(data['poses'])
    poses = poses.reshape(-1, 3)  # reshape 为 (24, 3) 的形状
    rotation_matrices = [axangle2mat(pose[:3], np.linalg.norm(pose)) for pose in poses]
    
    # 形状参数
    shapes = np.array(data['shapes'])
    
    return Rh, Th, rotation_matrices, shapes

data = {
    "id": 0,
    "Rh": [[-3.041, -0.204, 0.182]],
    "Th": [[-0.485, -0.096, 3.768]],
    "poses": [[-0.919, 0.233, 0.142, 0.403, -0.399, -0.126, 0.227, 0.178, 0.013, 1.447, 0.904, -0.451, -0.673, 0.088, -0.210, -0.082, -0.169, -0.027, 0.010, -0.052, -0.101, 0.363, -0.340, 0.585, 0.063, -0.035, 0.042, -0.384, 0.285, 0.001, -0.288, 0.580, -0.906, -0.233, 0.317, -0.165, -0.158, -0.460, -0.187, 0.013, 0.116, 0.262, 0.197, 0.474, 0.135, -0.346, -0.848, -0.575, 0.123, -0.046, 0.650, -0.074, -0.849, 0.103, -0.176, 0.366, 0.119, -0.085, -0.148, 0.375, 0.215, 0.045, -0.435, 0.227, -0.197, -0.141, -0.144, 0.223, 0.346]],
    "shapes": [[0.493, 0.502, -0.499, 0.506, -0.353, 0.001, -0.003, 0.403, 0.000, 0.250]]
}

Rh, Th, rotation_matrices, shapes = process_smpl_data(data)

# 打印旋转矩阵
for i, rotation_matrix in enumerate(rotation_matrices):
    print(f"Rotation matrix for joint {i}:")
    print(rotation_matrix)
    print()
