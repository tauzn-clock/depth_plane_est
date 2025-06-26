import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'Depth-Anything-V2/metric_depth'))

import torch
import cv2
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2

def run_metric3d(rgb_path):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vitl' # or 'vits', 'vitb'
    dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20 # 20 for indoor model, 80 for outdoor model

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load('/scratchdata/depth_anything_v2_metric_hypersim_vitl.pth'))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()

    raw_img = cv2.imread(rgb_path)
    depth = model.infer_image(raw_img) # HxW depth map in meters in numpy

    print(depth.max(), depth.min())

    return depth


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rgb_path = '/scratchdata/processed/stair4/rgb/0.png'
    output_depth = run_metric3d(rgb_path)

    plt.imsave('depth.png', output_depth)
