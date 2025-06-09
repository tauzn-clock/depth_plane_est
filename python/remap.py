import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from depth_plane_est.get_planes import get_planes

DATA_DIR = "/scratchdata/processed/outdoor_highres_unfiltered"

with open(os.path.join(DATA_DIR, "camera_info.json"), "r") as f:
    data = json.load(f)
    INTRINSICS = [data["K"][0], data["K"][4], data["K"][2], data["K"][5]]
    
print("INTRINSICS:", INTRINSICS)

INDEX = 390

rgb = Image.open(os.path.join(DATA_DIR, "rgb", f"{INDEX}.png")).convert('RGB')
depth = Image.open(os.path.join(DATA_DIR, "depth", f"{INDEX}.png"))

rgb = np.array(rgb)
depth = np.array(depth) / 1000.0  # Convert to meters and scale

from depth_plane_est.process_depth import get_normal_adj, get_normal_svd
normal = get_normal_svd(depth, INTRINSICS, kernel_size=21)

plt.imsave("normal.png", (normal + 1) / 2.0)

from run_metric3d import run_metric3d
metric3d_depth, metric3d_normal = run_metric3d(rgb)
print(metric3d_depth.max(), metric3d_depth.min())
metric3d_normal = metric3d_normal * np.where(metric3d_normal[:,:,2]> 0, 1, -1)[..., np.newaxis]

plt.imsave("metric3d_depth.png", metric3d_depth)
plt.imsave("metric3d_normal.png", (metric3d_normal + 1) / 2.0)

from linear_rescale import linear_rescale, plot_rescale, get_metrics

# All points
mask = depth > 0
m, b = linear_rescale(depth[mask], metric3d_depth[mask])
print(f"Linear rescale (All): m = {m}, b = {b}")

plot_rescale(depth[depth>0], metric3d_depth[depth>0], m, b, f"{DATA_DIR}/all_best_fit.png")
get_metrics(depth[depth>0], metric3d_depth[depth>0], m, b,f"{DATA_DIR}/all_histogram.png")

# Points between 1 and 5 m
mask = (depth > 1) & (depth < 5)
m, b = linear_rescale(depth[mask], metric3d_depth[mask])
print(f"Linear rescale (1-5m): m = {m}, b = {b}")

plot_rescale(depth[depth>0], metric3d_depth[depth>0], m, b, f"{DATA_DIR}/interval_best_fit.png")
get_metrics(depth[depth>0], metric3d_depth[depth>0], m, b, f"{DATA_DIR}/interval_histogram.png")

#RANSAC
from linear_rescale import linear_rescale_ransac
mask = depth > 0
m, b = linear_rescale_ransac(depth[mask], metric3d_depth[mask])
print(f"RANSAC rescale: m = {m}, b = {b}")

plot_rescale(depth[depth>0], metric3d_depth[depth>0], m, b, f"{DATA_DIR}/ransac_best_fit.png")
get_metrics(depth[depth>0], metric3d_depth[depth>0], m, b, f"{DATA_DIR}/ransac_histogram.png")

# Planes
mask, _ = get_planes(depth, normal, INTRINSICS, 5, 0.02)
mask = mask > 0

m, b = linear_rescale(depth[mask], metric3d_depth[mask])
print(f"Linear rescale (Planes): m = {m}, b = {b}")

plot_rescale(depth[depth>0], metric3d_depth[depth>0], m, b, f"{DATA_DIR}/planes_best_fit.png")
get_metrics(depth[depth>0], metric3d_depth[depth>0], m, b, f"{DATA_DIR}/planes_histogram.png")