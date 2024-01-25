import numpy as np
from transforms3d.axangles import axangle2mat
import os
import json

# Define input and output directories
input_dir = './output/sv1p/smpl/'
output_dir = './output/sv1p/rot3x3/'

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
    
    return rotation_matrices,Rh, Th, shapes


# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        # Load SMPL parameters
        with open(os.path.join(input_dir, filename), 'r') as f:
            params_list = json.load(f)

        result_list = {
            'rot3x3': []
        }
        for data in params_list:
            rotation_matrices,Rh, Th, shapes = process_smpl_data(data)
            for i, rotation_matrix in enumerate(rotation_matrices):
                result_list['rot3x3'].append(rotation_matrix.flatten().tolist())

        # Save the results to a new JSON file
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(result_list, f)