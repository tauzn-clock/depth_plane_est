import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt

from depth_to_pcd import depth_to_pcd
from get_normal import get_normal

from hsv import hsv_img

DIR = "/home/daoxin/scratchdata/processed/stair4_filtered"
with open(os.path.join(DIR, "camera_info.json"), "r") as f:
    camera_info = json.load(f)
INTRINSICS = camera_info["P"]
print(INTRINSICS)

# Open csv
odom = []
with open(os.path.join(DIR, "pose.csv"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(",")
        odom.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])])
odom = np.array(odom)

for INDEX in range(0,1000):
    # load image as tensor in range [0, 1] with shape [C, H, W]
    image = Image.open(os.path.join(DIR,f"rgb/{INDEX}.png"))
    image = np.array(image)
    depth = Image.open(os.path.join(DIR, f"depth/{INDEX}.png"))
    depth = np.array(depth)/1000

    # Convert depth to point cloud

    points, index = depth_to_pcd(depth, INTRINSICS)

    # Find distnace of pts

    normal = [odom[INDEX, 0], odom[INDEX, 1], odom[INDEX, 2]]
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    print(normal)

    # Find img normal
    img_normal = get_normal(depth, INTRINSICS)
    img_normal = img_normal.reshape(-1, 3)

    if True:
        img_normal_rgb = (img_normal + 1)/2 * 255
        img_normal_rgb = img_normal_rgb.astype(np.uint8)
        H, W = depth.shape
        img_normal_rgb = img_normal_rgb.reshape(H, W, 3)
        plt.imsave("normal.png", img_normal_rgb)

    # Get distances

    dot1 = np.dot(img_normal, normal)
    dot2 = np.dot(img_normal, -normal)
    dot1 = dot1.reshape(-1,1)
    dot2 = dot2.reshape(-1,1)

    angle_dist = np.concatenate((dot1, dot2), axis=1)
    angle_dist = np.max(angle_dist, axis=1)

    scalar_dist = np.dot(points, normal)
    scalar_dist[angle_dist < 0.9] = 0
    scalar_dist[points[:, 2] == 0] = 0

    if True:
        # Plot histogram
        fig, ax = plt.subplots()
        ax.hist(scalar_dist[scalar_dist!=0], bins=1000)
        plt.xlabel("Distance")
        plt.ylabel("Count")
        plt.title("Histogram of Distance")
        fig.savefig("histogram.png")

    K = 8
    
    index = index[scalar_dist != 0]
    scalar_dist = scalar_dist[scalar_dist != 0]
    if (len(scalar_dist)==0):
        continue

    bins = np.arange(scalar_dist.min(), scalar_dist.max(), 0.01)
    hist, bin_edges = np.histogram(scalar_dist, bins=bins)

    kernel_size = 11
    kernel = [-kernel_size//2 + 1 + i for i in range(kernel_size)]

    group_size = 5
    group = [-group_size//2 + 1 + i for i in range(group_size)]

    # Dilation of histogram
    dilation_hist = np.pad(hist, (kernel_size//2, kernel_size//2))
    dilation_hist = [np.roll(dilation_hist, i) for i in kernel]
    dilation_hist = np.array(dilation_hist)
    dilation_hist = np.max(dilation_hist, axis=0)[kernel_size//2:-kernel_size//2+1]
    
    # Get index where dilation_hist is equal to hist
    candidate_peak = np.where(dilation_hist == hist)[0]

    local_total = np.pad(hist, (group_size//2, group_size//2))
    local_total = [np.roll(local_total, i) for i in group]
    local_total = np.array(local_total)
    local_total = np.sum(local_total, axis=0)[group_size//2:-group_size//2+1]

    # Get index of the 10 largest values in local_total
    best_peaks = np.argsort(local_total[candidate_peak])[-4:]
    best_peaks_index = candidate_peak[best_peaks]
    
    mask_2d = np.zeros_like(depth, dtype=np.uint8)
    for i in range(len(best_peaks_index)):
        for j in range(len(scalar_dist)):
            if (best_peaks_index[i] - group_size//2) >=0:
                lower_bound = bin_edges[best_peaks_index[i]-group_size//2]
            else:
                lower_bound = bin_edges[0]
            if (best_peaks_index[i] + 1 + group_size//2) < len(bin_edges):
                upper_bound = bin_edges[best_peaks_index[i] + 1 + group_size//2]
            else:
                upper_bound = bin_edges[-1]
            if scalar_dist[j] > lower_bound and scalar_dist[j] < upper_bound:
                mask_2d[index[j, 0], index[j, 1]] = i + 1
    print(mask_2d.max())


    """
    # Randomly sample K points
    idx = np.random.choice(len(scalar_dist), K, replace=False)
    means = scalar_dist[idx]

    for _ in range(100):
        dist = np.abs(scalar_dist[:, None] - means[None, :])
        dist = abs(dist)
        mask = np.argmin(dist, axis=1)
        means = np.zeros((K, 1))
        for i in range(K):
            means[i] = np.mean(scalar_dist[mask == i])
        means = means.reshape(-1)

    mask_2d = np.zeros_like(depth, dtype=np.uint8)
    for i in range(len(mask)):
        mask_2d[index[i, 0], index[i, 1]] = mask[i] + 1
    """
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
