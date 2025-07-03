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

from scipy.spatial.transform import Rotation as R

DIR = "/home/daoxin/scratchdata/processed/short"
with open(os.path.join(DIR, "camera_info.json"), "r") as f:
    camera_info = json.load(f)
INTRINSICS = camera_info["P"]
print(INTRINSICS)

USE_MEASURED = True
USE_ORIENTATION = True

model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True).cuda() 
model = model.cuda() if torch.cuda.is_available() else model

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
        img_normal = get_normal(depth, INTRINSICS)
    else:
        depth = Image.open(os.path.join(DIR, f"depth/{INDEX}.png"))
        depth = np.array(depth)/1000
        _, img_normal = metric3d(model, image)

    # Convert depth to point cloud
    W, H = depth.shape
    points, index = depth_to_pcd(depth.flatten(), INTRINSICS, H, W)
    
    # Find distnace of pts
    if USE_ORIENTATION:
        print()

        #grav_normal = odom[INDEX, 0:3]
        #grav_normal = grav_normal / np.linalg.norm(grav_normal)
        #print(grav_normal)

        orientation_quat = [odom[INDEX, 7], odom[INDEX, 8], odom[INDEX, 9], odom[INDEX, 6]]
        orientation = R.from_quat(orientation_quat).as_matrix()
    
        grav_normal = np.array([0, 0, -1])
        grav_normal = orientation.T @ grav_normal
        grav_normal = grav_normal / np.linalg.norm(grav_normal)
        grav_normal = np.array([grav_normal[1], grav_normal[2], grav_normal[0]]) # Transform to camera frame, unique for this tf
        print(grav_normal)
    else:
        grav_normal = odom[INDEX, 0:3]
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

    if True:
        """
        # Plot normal in 3D
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        bound = 0.9
        ax.scatter(img_normal[(np.dot(img_normal, grav_normal) < bound) & (points[:,2]!=0), 0], img_normal[(np.dot(img_normal, grav_normal) < bound) & (points[:,2]!=0), 1], img_normal[(np.dot(img_normal, grav_normal) < bound) & (points[:,2]!=0), 2], marker='o', s=1, c='b')
        ax.scatter(img_normal[(np.dot(img_normal, grav_normal) > bound) & (points[:,2]!=0), 0], img_normal[(np.dot(img_normal, grav_normal) > bound) & (points[:,2]!=0), 1], img_normal[(np.dot(img_normal, grav_normal) > bound) & (points[:,2]!=0), 2], marker='o', s=1, c='r')
        fig.savefig("sphere.png")
        """

        img_normal_rgb = (img_normal + 1)/2 * 255
        img_normal_rgb = img_normal_rgb.astype(np.uint8)
        H, W = depth.shape
        img_normal_rgb = img_normal_rgb.reshape(H, W, 3)
        plt.imsave("normal.png", img_normal_rgb)
    
    dot_bound = 0.9
    correction_iteration = 10
    grav_normal = gravity_correction(grav_normal,img_normal, points.reshape(-1,3), dot_bound, correction_iteration)

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

    mask = get_mask(grav_normal, img_normal, points.reshape(-1,3), dot_bound, kernel_size, cluster_size)
    mask_2d = mask.reshape(H, W)

    if True:
        # Plot mask
        fig, ax = plt.subplots()
        ax.imshow(hsv_img(mask_2d))
        plt.axis('off')
        # Save mask
        plt.savefig("mask.png")

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.imshow(hsv_img(mask_2d), alpha=0.5, cmap="hsv")
        plt.axis('off')
        # Save depth
        plt.savefig(os.path.join(DIR, f"test/{INDEX}.png"), bbox_inches='tight', pad_inches=0)


