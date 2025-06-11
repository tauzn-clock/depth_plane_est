import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DATA_DIR = "/scratchdata/processed/alcove2"

with open(os.path.join(DATA_DIR, "camera_info.json"), "r") as f:
    data = json.load(f)
    INTRINSICS = [data["K"][0], data["K"][4], data["K"][2], data["K"][5]]
    
print("INTRINSICS:", INTRINSICS)

INDEX = 0

rgb = Image.open(os.path.join(DATA_DIR, "rgb", f"{INDEX}.png")).convert('RGB')
depth = Image.open(os.path.join(DATA_DIR, "depth", f"{INDEX}.png"))

rgb = np.array(rgb)
depth = np.array(depth) / 1000.0  # Convert to meters and scale
depth = depth #/ 10.0 #depth.max()

from depth_plane_est.process_depth import get_normal_adj, get_normal_svd
normal = get_normal_svd(depth, INTRINSICS, kernel_size=21)

plt.imsave("normal.png", (normal + 1) / 2.0)

if True:
    from run_metric3d import run_metric3d
    metric3d_depth, metric3d_normal = run_metric3d(rgb)
    print(metric3d_depth.max(), metric3d_depth.min())
    metric3d_normal = metric3d_normal * np.where(metric3d_normal[:,:,2]> 0, 1, -1)[..., np.newaxis]

    plt.imsave("metric3d_depth.png", metric3d_depth)
    plt.imsave("metric3d_normal.png", (metric3d_normal + 1) / 2.0)

    #depth = metric3d_depth
    #normal = metric3d_normal

# Rescale method
from linear_rescale import linear_rescale, plot_rescale, get_metrics

#RANSAC
from linear_rescale import linear_rescale_ransac
mask = depth > 0
m, b = linear_rescale_ransac(depth[mask], metric3d_depth[mask])
print(f"RANSAC rescale: m = {m}, b = {b}")

def check_range(depth, rescale, bound):
    print(depth.shape)
    diff = abs(depth - rescale)
    mask = np.zeros_like(diff, dtype=np.int8)
    mask[depth != 0] = 1
    mask[diff < bound] = 2
    return mask

def combine_mask_with_rgb(rgb, mask):
    combined = np.zeros((*mask.shape, 3), dtype=np.uint8)
    combined[mask == 0] = [0, 0, 0]  # Black for invalid points
    combined[mask == 1] = 0.5 * (rgb[mask == 1] + [255, 0, 0])  # Red for points outside the range
    combined[mask == 2] = 0.5 * (rgb[mask == 2] + [0, 255, 0])  # Green for points within the range
    return combined

mask = check_range(depth, m*metric3d_depth + b, 0.01)
img_1 = combine_mask_with_rgb(rgb, mask)
ratio_1 = (100 * np.sum(mask == 2) / np.sum(mask != 0))

mask = check_range(depth, m*metric3d_depth + b, 0.02)
img_2 = combine_mask_with_rgb(rgb, mask)
ratio_2 = (100 * np.sum(mask == 2) / np.sum(mask != 0))

mask = check_range(depth, m*metric3d_depth + b, 0.05)
img_3 = combine_mask_with_rgb(rgb, mask)
ratio_3 = (100 * np.sum(mask == 2) / np.sum(mask != 0))

mask = check_range(depth, m*metric3d_depth + b, 0.1)
img_4 = combine_mask_with_rgb(rgb, mask)
plt.imsave("mask_10.png", img_4)
ratio_4 = (100 * np.sum(mask == 2) / np.sum(mask != 0))

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].imshow(img_1)
ax[0, 0].set_title(f"Bound: 0.01, Ratio: {ratio_1:.2f}%")
ax[0, 1].imshow(img_2)
ax[0, 1].set_title(f"Bound: 0.02, Ratio: {ratio_2:.2f}%")
ax[1, 0].imshow(img_3)
ax[1, 0].set_title(f"Bound: 0.05, Ratio: {ratio_3:.2f}%")
ax[1, 1].imshow(img_4)
ax[1, 1].set_title(f"Bound: 0.1, Ratio: {ratio_4:.2f}%")
ax[0, 0].axis('off')
ax[0, 1].axis('off')
ax[1, 0].axis('off')
ax[1, 1].axis('off')
plt.tight_layout()
plt.savefig("rescale_comparison.png")