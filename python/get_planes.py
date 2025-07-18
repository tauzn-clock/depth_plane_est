import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DATA_DIR = "/scratchdata/processed/outdoor_highres"

with open(os.path.join(DATA_DIR, "camera_info.json"), "r") as f:
    data = json.load(f)
    INTRINSICS = [data["K"][0], data["K"][4], data["K"][2], data["K"][5]]
    
print("INTRINSICS:", INTRINSICS)

INDEX = 349

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

    depth = metric3d_depth
    normal = metric3d_normal

from depth_plane_est.get_planes import get_planes
mask, param = get_planes(depth, normal, INTRINSICS, 5, 0.02)
print(mask.max())

from depth_plane_est.mask_to_hsv import mask_to_hsv
plt.imsave("mask.png", mask_to_hsv(mask))

from depth_plane_est.save_pcd import save_pcd
from depth_plane_est.process_depth import get_3d
save_pcd(rgb, get_3d(depth, INTRINSICS), f"{DATA_DIR}/output.ply")
save_pcd(mask_to_hsv(mask), get_3d(depth, INTRINSICS), f"{DATA_DIR}/mask.ply")

from depth_plane_est.save_pcd import save_planes
save_planes(get_3d(depth, INTRINSICS), mask, param, f"{DATA_DIR}/planes.ply")

from depth_plane_est.save_pcd import save_normal
save_normal(get_3d(depth, INTRINSICS), normal, f"{DATA_DIR}/normal.ply")