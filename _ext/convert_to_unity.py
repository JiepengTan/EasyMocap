# 安装 smplx

import os
import json
import smplx
import numpy as np
from smplx.lbs import lbs
from scipy.spatial.transform import Rotation as R

# Define SMPL model
model = smplx.create('./models', model_type='smplx')

print("succ load model")
# Define input and output directories
input_dir = './output/sv1p/smpl/'
output_dir = './output/smpl_parse/'

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        # Load SMPL parameters
        with open(os.path.join(input_dir, filename), 'r') as f:
            params_list = json.load(f)

        result_list = []
        for params in params_list:
            # Convert lists to numpy arrays
            print(params['poses'])
            poses = np.array(params['poses'][0])
            shapes = np.array(params['shapes'][0])
            Rh = np.array(params['Rh'][0])
            Th = np.array(params['Th'][0])

            # Get SMPL output
            output = model(body_pose=poses[3:], global_orient=poses[:3], betas=shapes)

            # Convert rotation matrices to Euler angles
            rotations = R.from_dcm(output.joint_rotation)
            euler_angles = rotations.as_euler('xyz')

            # Collect each joint's position and rotation
            result = {
                'id': params['id'],
                'joints': []
            }
            for i in range(output.joints.shape[0]):
                result['joints'].append({
                    'position': output.joints[i].tolist(),
                    'rotation': euler_angles[i].tolist()
                })
            result_list.append(result)

        # Save the results to a new JSON file
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(result_list, f)
