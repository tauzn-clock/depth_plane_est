from geocalib import GeoCalib
import torch
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt

from depth_to_pcd import depth_to_pcd
from get_normal import get_normal

from information_optimisation import plane_ransac
from hsv import hsv_img

DIR = "/home/daoxin/scratchdata/processed/stairs_up"
for INDEX in range(1,1000):
    with open(os.path.join(DIR, "camera_info.json"), "r") as f:
        camera_info = json.load(f)
    INTRINSICS = camera_info["P"]
    print(INTRINSICS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoCalib().to(device)

    # load image as tensor in range [0, 1] with shape [C, H, W]
    image = model.load_image(os.path.join(DIR,f"rgb/{INDEX}.png")).to(device)
    depth = Image.open(os.path.join(DIR, f"depth/{INDEX-1}.png"))
    depth = np.array(depth)/1000

    result = model.calibrate(image)

    # Convert depth to point cloud

    points, index = depth_to_pcd(depth, INTRINSICS)
    print(points.shape)
    print(index.shape)

    print(depth.max(), points[:, 2].max())

    # Find distnace of pts

    normal = [result["gravity"].x.item(), result["gravity"].y.item(), result["gravity"].z.item()]
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    print(normal)

    # Find img normal
    img_normal = get_normal(depth, INTRINSICS)
    print(img_normal.shape)

    print(img_normal.max(), img_normal.min())

    if False:
        img_normal_rgb = (img_normal + 1)/2 * 255
        img_normal_rgb = img_normal_rgb.astype(np.uint8)
        plt.imsave("normal.png", img_normal_rgb)


    img_normal = img_normal.reshape(-1, 3)

    dot1 = np.dot(img_normal, normal)
    dot2 = np.dot(img_normal, -normal)
    dot1 = dot1.reshape(-1,1)
    dot2 = dot2.reshape(-1,1)

    angle_dist = np.concatenate((dot1, dot2), axis=1)
    angle_dist = np.max(angle_dist, axis=1)

    scalar_dist = np.dot(points, normal)
    scalar_dist[angle_dist < 0.9] = 0
    print(scalar_dist.max(), scalar_dist.min())

    if True:
        # Plot histogram
        fig, ax = plt.subplots()
        ax.hist(scalar_dist, bins=100)
        plt.xlabel("Distance")
        plt.ylabel("Count")
        plt.title("Histogram of Distance")
        plt.show()

    """
    # Bin dist into bins of size 0.1
    bins = np.arange(scalar_dist.min(), scalar_dist.max(), 0.2)
    hist, bin_edges = np.histogram(scalar_dist, bins=bins)

    mask = np.zeros_like(depth)
    H, W = depth.shape
    cnt = 1
    for i in range(len(hist)):
        corresponding_index = index[(scalar_dist > bin_edges[i]) & (scalar_dist <= bin_edges[i+1]) & (scalar_dist != 0)]
        if len(corresponding_index) < 0.01 * H * W:
            continue
        mask[corresponding_index[:, 0], corresponding_index[:, 1]] = cnt
        cnt += 1
    """

    mask = np.zeros_like(angle_dist, dtype=bool)
    mask[angle_dist > 0.9] = True

    R = 10.0
    EPSILON = 0.001
    SIGMA = depth.flatten() * 0.01
    CONFIDENCE = 0.99
    INLIER_THRESHOLD = 0.2
    MAX_PLANE = 5

    information, mask, plane = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, mask,verbose=True,post_processing=False)
    
    min_idx = np.argmin(information)
    print("Information: ", information)
    print("Min Mask IDX: ", min_idx)
    for i in range(min_idx+1, MAX_PLANE+1):
        mask[mask == i] = 0
    mask = mask.reshape(depth.shape)
    print(mask.max(), mask.min())
    if True:
        # Plot mask
        fig, ax = plt.subplots()
        ax.imshow(hsv_img(mask))
        plt.axis('off')
        # Save mask
        plt.savefig("mask.png")

        fig, ax = plt.subplots()
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.imshow(hsv_img(mask), alpha=0.5, cmap="hsv")
        plt.axis('off')
        # Save depth
        plt.savefig(os.path.join(DIR, f"test/{INDEX}.png"), bbox_inches='tight', pad_inches=0)