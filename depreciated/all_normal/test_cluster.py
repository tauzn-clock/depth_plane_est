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

from scipy.spatial.transform import Rotation as R

DIR = "/home/daoxin/scratchdata/processed/stair4"
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

# Open csv
odom = []
with open(os.path.join(DIR, "pose.csv"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(",")
        odom.append([float(x) for x in line])
odom = np.array(odom)

for INDEX in range(0,1000):
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

    W, H = depth.shape

    img_normal = img_normal * np.where(img_normal[:,:,2] >=0, 1, -1).reshape(W, H, 1)

    points, index = depth_to_pcd(depth.flatten(), INTRINSICS, H, W)
    #img_dist = img_normal * points.reshape(W, H, 3)
    #img_dist = np.sum(img_dist, axis=2).reshape(W, H, 1)
    #print(img_dist.shape, img_dist.max(), img_dist.min())
    #img_normal = img_normal * np.where(img_dist >=0, 1, -1)
    #print(np.where(img_dist >=0, 1, -1).shape)
    
    if True: plt.imsave("normal.png", (img_normal+1)/2)

    img_normal_angle = np.zeros((W, H, 2))
    img_normal_angle[:, :, 0] = np.arctan(img_normal[:,:,0]/(img_normal[:,:,2]+1e-15))
    img_normal_angle[:, :, 1] = np.arctan(img_normal[:,:,1]/(img_normal[:,:,2]+1e-15))

    print(img_normal_angle[:,:,0].max(), img_normal_angle[:,:,0].min())
    print(img_normal_angle[:,:,1].max(), img_normal_angle[:,:,1].min())

    print(img_normal_angle[100,100])

    angle_cluster = np.zeros((ANGLE_INCREMENT, ANGLE_INCREMENT))
    for i in range(0, ANGLE_INCREMENT):
        for j in range(0, ANGLE_INCREMENT):
            angle_cluster[i, j] = np.sum((img_normal_angle[:, :, 0] >= (i-ANGLE_INCREMENT/2)*np.pi/ANGLE_INCREMENT ) 
                                         & (img_normal_angle[:, :, 0] < (i+1-ANGLE_INCREMENT/2)*np.pi/ANGLE_INCREMENT)
                                         & (img_normal_angle[:, :, 1] >= (j-ANGLE_INCREMENT/2)*np.pi/ANGLE_INCREMENT)
                                         & (img_normal_angle[:, :, 1] < (j+1-ANGLE_INCREMENT/2)*np.pi/ANGLE_INCREMENT)
                                         & (depth != 0))

    dillation = np.zeros((ANGLE_INCREMENT, ANGLE_INCREMENT))
    angle_cluster = np.pad(angle_cluster, ((KERNEL_2D//2, KERNEL_2D//2), (KERNEL_2D//2, KERNEL_2D//2)), mode='wrap')
    for i in range(ANGLE_INCREMENT):
        for j in range(ANGLE_INCREMENT):
            dillation[i, j] = np.max(angle_cluster[i:i+KERNEL_2D, j:j+KERNEL_2D])

    angle_cluster = angle_cluster[KERNEL_2D//2:-KERNEL_2D//2+1, KERNEL_2D//2:-KERNEL_2D//2+1]

    if True:
        plt.imsave("angle_cluster.png", angle_cluster)

    mask = np.zeros_like(depth, dtype=np.uint8)

    # Find index where angle_cluster is equal to dillation
    index = np.where(angle_cluster == dillation)

    # Within the index, find the 3 largest clusters
    best_peaks = np.argsort(angle_cluster[index])[-3:]
    best_peaks_index = np.array([index[0][best_peaks], index[1][best_peaks]]).T

    for i in range(best_peaks_index.shape[0]):
        print(angle_cluster[best_peaks_index[i][0], best_peaks_index[i][1]])
        angle_x = (best_peaks_index[i][0] - ANGLE_INCREMENT//2) * np.pi / ANGLE_INCREMENT
        angle_y = (best_peaks_index[i][1] - ANGLE_INCREMENT//2) * np.pi / ANGLE_INCREMENT
        
        grav_normal = np.array([np.tan(angle_x), np.tan(angle_y), 1])
        grav_normal = grav_normal / np.linalg.norm(grav_normal)

        img_normal_pos = img_normal.reshape(-1, 3)
        img_normal_neg = -img_normal_pos
        dot1 = np.dot(img_normal_pos, grav_normal).reshape(-1, 1)
        dot2 = np.dot(img_normal_neg, grav_normal).reshape(-1, 1)
        
        dot = np.concatenate((dot1, dot2), axis=1)
        normal_index = np.argmax(dot, axis=1)
        img_normal = np.zeros_like(img_normal_pos)
        img_normal[normal_index == 0] = img_normal_pos[normal_index == 0]
        img_normal[normal_index == 1] = img_normal_neg[normal_index == 1]

        dot_bound = 0.9
        correction_iteration = 10
        grav_normal = gravity_correction(grav_normal,img_normal.reshape(-1,3), points.reshape(-1,3), dot_bound, correction_iteration)

        if True:
            dot1 = np.dot(img_normal, grav_normal).reshape(-1,1)
            dot2 = np.dot(img_normal, -grav_normal).reshape(-1,1)

            angle_dist = np.concatenate((dot1, dot2), axis=1)
            angle_dist = np.max(angle_dist, axis=1)
            scalar_dist = np.dot(points.reshape(-1,3), grav_normal)
            scalar_dist[angle_dist < dot_bound] = 0
            scalar_dist[points.reshape(-1,3)[:, 2] == 0] = 0

            # Plot histogram
            fig, ax = plt.subplots()
            ax.hist(scalar_dist[scalar_dist!=0], bins=1000)
            plt.xlabel("Distance")
            plt.ylabel("Count")
            plt.title("Histogram of Distance")
            fig.savefig("./histogram.png")

        kernel_size = 11
        cluster_size = 11

        mask_2d = get_mask(grav_normal, img_normal.reshape(-1,3), points.reshape(-1,3), dot_bound, kernel_size, cluster_size)
        mask_2d = mask_2d.reshape(W, H)
        cur_plane = mask.max()
        mask = np.where(mask_2d != 0, mask_2d + cur_plane, mask)

    print(mask.max(), mask.min())

    if True:
        # Plot mask
        fig, ax = plt.subplots()
        ax.imshow(hsv_img(mask))
        plt.axis('off')
        # Save mask
        plt.savefig("mask.png")

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.imshow(hsv_img(mask), alpha=0.5, cmap="hsv")
        plt.axis('off')
        # Save depth
        plt.savefig(os.path.join(DIR, f"test/{INDEX}.png"), bbox_inches='tight', pad_inches=0)
