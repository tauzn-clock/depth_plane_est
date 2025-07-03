import sys
sys.path.append("/GeoCalib")

import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DIR = "/scratchdata/processed/long"

with open(os.path.join(DIR, "camera_info.json"), "r") as f:
    camera_info = json.load(f)
INTRINSICS = camera_info["P"]
print(INTRINSICS)

INDEX = 0

image = Image.open(os.path.join(DIR,f"rgb/{INDEX}.png"))
image = np.array(image)

depth = Image.open(os.path.join(DIR, f"depth/{INDEX}.png"))
depth = np.array(depth)/1000
print(depth.max())

plt.imsave("depth.png", depth)

from Metric3D.run_metric3d import run_metric3d

metric3d_depth, metric3d_normal = run_metric3d(image)
print(metric3d_depth.max())

plt.imsave("metric3d.png", metric3d_depth)



def get_metrics(depth, est):
    
    est = est.flatten()
    depth = depth.flatten()
    
    est = est[depth > 0]  # Filter out invalid depth values
    depth = depth[depth > 0]  # Filter out invalid depth values
    
    diff = np.abs(depth - est)

    # Save histogram
    fig, ax = plt.subplots()
    ax.hist(diff, bins=100)
    ax.set_xlabel("Absolute error (m)")
    ax.set_ylabel("Frequency")

    plt.savefig("histogram.png")

    threshold = np.maximum((depth / est), (est / depth))
    delta1 = (threshold < 1.25).mean()
    print(f"Delta1: {delta1:.4f}")
    
    rmse = np.sqrt(np.mean((depth - est) ** 2))
    print(f"RMSE: {rmse:.4f} m")
    
    percentile_95 = np.percentile(diff, 95)
    print(f"95th Percentile Error: {percentile_95:.4f} m")

from utils.depth_to_pcd import depth_to_pcd
from utils.save_pcd import save_pcd
W,H = depth.shape

pcd,_ = depth_to_pcd(depth.flatten(), INTRINSICS, H, W)
save_pcd(image, pcd, os.path.join(DIR, f"depth.ply"))

pcd,_ = depth_to_pcd(metric3d_depth.flatten(), INTRINSICS, H, W)
save_pcd(image, pcd, os.path.join(DIR, f"metric3d.ply"))

get_metrics(depth, metric3d_depth)

from linear_rescale import linear_rescale

rescale = linear_rescale(depth, metric3d_depth)

pcd,_ = depth_to_pcd(rescale.flatten(), INTRINSICS, H, W)
save_pcd(image, pcd, os.path.join(DIR, f"linear_rescale.ply"))

get_metrics(depth, rescale)

from inverse_linear_rescale import inverse_linear_rescale

rescale = inverse_linear_rescale(depth, metric3d_depth)

pcd,_ = depth_to_pcd(rescale.flatten(), INTRINSICS, H, W)
save_pcd(image, pcd, os.path.join(DIR, f"inverse_linear_rescale.ply"))

get_metrics(depth, rescale)

from utils.scaled_rigid_transform import scaled_rigid_transform

gt_3d,_ = depth_to_pcd(depth.flatten(), INTRINSICS, H, W)
est_3d,_ = depth_to_pcd(metric3d_depth.flatten(), INTRINSICS, H, W)

s, R, t = scaled_rigid_transform(est_3d[gt_3d[:,2]!=0], gt_3d[gt_3d[:,2]!=0])
print("Scale:", s)
print("Rotation:", R)
print("Translation:", t)

est_3d = s * np.dot(est_3d, R.T) + t
save_pcd(image, est_3d, os.path.join(DIR, f"scaled_rigid_transform.ply"))

gt_3d_z = gt_3d[:, 2]
est_3d_z = est_3d[:, 2]

est_3d_z = est_3d_z[gt_3d_z != 0]
gt_3d_z = gt_3d_z[gt_3d_z != 0]

fig, ax = plt.subplots()
ax.scatter(est_3d_z, gt_3d_z, s=1, alpha=0.5, label='Data Points')
ax.set_xlabel('Estimated Depth (m)')
ax.set_ylabel('Ground Truth Depth (m)')

plt.savefig("rigid_transform.png")

get_metrics(gt_3d_z, est_3d_z)