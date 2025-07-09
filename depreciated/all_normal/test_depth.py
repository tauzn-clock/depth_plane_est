import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import torch
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

from utils.depth_to_pcd import depth_to_pcd
from utils.get_normal import get_normal
from utils.gravity_correction import gravity_correction
from utils.get_mask import get_mask
from utils.hsv import hsv_img
from utils.metric3d import metric3d
from utils.test_depth import parabolic, flat
from utils.get_planes import get_planes

from Metric3D.run_metric3d import run_metric3d

#from scipy.spatial.transform import Rotation as R

DIR = "/scratchdata/processed/stair4"

with open(os.path.join(DIR, "camera_info.json"), "r") as f:
    camera_info = json.load(f)
INTRINSICS = camera_info["P"]
print(INTRINSICS)

USE_MEASURED = True
USE_ORIENTATION = True
ANGLE_INCREMENT = 41
KERNEL_2D = 5

#model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True).cuda() 
#model = model.cuda() if torch.cuda.is_available() else model

## Open csv
#odom = []
#with open(os.path.join(DIR, "pose.csv"), "r") as f:
#    lines = f.readlines()
#    for line in lines:
#        line = line.strip().split(",")
#        odom.append([float(x) for x in line])
#odom = np.array(odom)

for INDEX in range(20,1000):
    # load image as tensor in range [0, 1] with shape [C, H, W]
    image = Image.open(os.path.join(DIR,f"rgb/{INDEX}.png"))
    image = np.array(image)
    if USE_MEASURED:
        depth = Image.open(os.path.join(DIR, f"depth/{INDEX}.png"))
        depth = np.array(depth)/1000
        W, H = depth.shape
        #depth = flat(H,W)
        img_normal = get_normal(depth, INTRINSICS)
    else:
        depth = Image.open(os.path.join(DIR, f"depth/{INDEX}.png"))
        depth = np.array(depth)/1000
        #_, img_normal = metric3d(model, image)

    print("Depth", depth.max())

    metric3d_depth, metric3d_normal = run_metric3d(image)

    plt.imsave('metric3d_depth.png', metric3d_depth)
    plt.imsave('metric3d_normal.png', (metric3d_normal+1)/2)
    
    mask = get_planes(depth, INTRINSICS, ANGLE_CLUSTER = 3, RATIO_SIZE = 0.02)

    def rescale(measured, est, mask):
        input_measured = measured[mask == 1]
        input_est = est[mask == 1]

        input_est = input_est[input_measured != 0]
        input_measured = input_measured[input_measured != 0]

        # Find line of best fit
        A = np.vstack([input_est, np.ones(len(input_est))]).T
        m, c = np.linalg.lstsq(A, input_measured, rcond=None)[0]

        return est * m + c
    
    rescaled = rescale(depth, metric3d_depth, mask)
    print(rescaled.max(), rescaled.min())
    print(depth.max(), depth.min())

    plt.imsave('rescaled.png', rescaled)
    plt.imsave('depth.png', depth)
    diff = rescaled - depth
    #diff[depth > 4] = 0
    diff[depth == 0] = 0
    diff = abs(diff)
    print(diff.max(), diff.min())
    plt.imsave('diff.png', diff)

    # Histogram of diff
    fig, ax = plt.subplots()
    ax.hist(diff[diff!=0], bins=1000)
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.title("Histogram of Distance")
    fig.savefig("./diff_histogram.png")

    if True:
        # Plot mask
        fig, ax = plt.subplots()
        ax.imshow(hsv_img(mask))
        plt.axis('off')
        # Save mask
        plt.savefig("mask.png")
        
        #fig, ax = plt.subplots()
        #ax.imshow(image)
        #ax.imshow(hsv_img(mask), alpha=0.5, cmap="hsv")
        #plt.axis('off')
        # Save depth
        #plt.savefig(os.path.join(DIR, f"test/{INDEX}.png"), bbox_inches='tight', pad_inches=0)


    from save_pcd import save_pcd

    print(depth.shape, depth.max())
    print(rescaled.shape, rescaled.max())
    print(metric3d_depth.shape, metric3d_depth.max())

    pcd,_ = depth_to_pcd(rescaled.flatten(), INTRINSICS, H, W)
    save_pcd(image, pcd, os.path.join(DIR, f"rescaled.ply"))
    pcd,_ = depth_to_pcd(metric3d_depth.flatten(), INTRINSICS, H, W)
    save_pcd(image, pcd, os.path.join(DIR, f"metric3d.ply"))
    pcd,_ = depth_to_pcd(depth.flatten(), INTRINSICS, H, W)
    save_pcd(image, pcd, os.path.join(DIR, f"depth.ply"))

    # Scaled rigid transform

    from utils.scaled_rigid_transform import scaled_rigid_transform

    gt_3d = depth_to_pcd(depth.flatten(), INTRINSICS, H, W)[0]
    est_3d = depth_to_pcd(metric3d_depth.flatten(), INTRINSICS, H, W)[0]
    print(gt_3d.shape, est_3d.shape)
    
    s, R, t = scaled_rigid_transform(est_3d[gt_3d[:,2]!=0], gt_3d[gt_3d[:,2]!=0])
    print("Scale:", s)
    print("Rotation:", R)
    print("Translation:", t)

    est_3d = s * np.dot(est_3d, R.T) + t
    save_pcd(image, est_3d, os.path.join(DIR, f"scaled_rigid_transform.ply"))

    exit()