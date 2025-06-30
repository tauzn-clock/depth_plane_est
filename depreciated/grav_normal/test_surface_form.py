import os
from PIL import Image
import json
import numpy as np
from matplotlib import pyplot as plt

from depth_to_pcd import depth_to_pcd


DATA_DIR = "/scratchdata/processed/stairs_down"

with open(os.path.join(DATA_DIR, "camera_info.json"), "r") as f:
    _data = json.load(f)
    INTRINSICS = _data['K']
    W = _data['width']
    H = _data['height']

INDEX = 0

rgb = Image.open(os.path.join(DATA_DIR, "rgb", f"{INDEX}.png")).convert("RGB")
rgb = np.array(rgb)
print(rgb.shape, rgb.max())
depth = Image.open(os.path.join(DATA_DIR, "depth", f"{INDEX}.png")).convert("I;16")
depth = np.array(depth) / 1000.0

plt.imsave("depth.png", depth)

pcd, _= depth_to_pcd(depth.flatten(), INTRINSICS, W, H)
pcd = pcd.reshape(H, W, 3)

"""
# Inject dummy pcd
for i in range(pcd.shape[0]):
    for j in range(pcd.shape[1]):
        pcd[i,j,0] = (i-0.5* pcd.shape[0]) / 10
        pcd[i,j,1] = (j-0.5* pcd.shape[1]) / 10
        pcd[i,j,2] = (pcd[i,j,0]**2 + pcd[i,j,1]**2) 
"""

def DX(vec):
    # Pad points along the edges
    vec = np.pad(vec, ((1, 1), (1, 1), (0, 0)))
    dx = (vec[1:-1, 2:, :] - vec[1:-1, :-2, :])
    dx = dx / np.stack(((dx[:,:,1] + 1e-16),)*3, axis=-1)
    return dx

def DY(vec):
    vec = np.pad(vec, ((1, 1), (1, 1), (0, 0)))
    dy = (vec[2:, 1:-1, :] - vec[:-2, 1:-1, :])
    dy = dy / np.stack(((dy[:,:,0] + 1e-16),)*3, axis=-1)
    return dy

def get_G(pcd):
    # Pad points along the edges
    dx = DX(pcd)
    dy = DY(pcd)
    
    G_xx = np.sum(dx * dx, axis=2)
    G_yy = np.sum(dy * dy, axis=2)
    G_xy = np.sum(dx * dy, axis=2)

    print(G_xx.max(), G_xx.min())
    print(G_yy.max(), G_yy.min())
    print(G_xy.max(), G_xy.min())

    plt.imsave("G_xx.png", G_xx)
    plt.imsave("G_yy.png", G_yy)
    plt.imsave("G_xy.png", G_xy)

    det = G_xx * G_yy - G_xy * G_xy
    plt.imsave("det.png", det[1:-1, 1:-1])
    print(det.max(),det.min())

    return None

get_G(pcd)

def get_L(pcd):
    pcd = np.pad(pcd, ((1, 1), (1, 1), (0, 0)))
    dx = pcd[1:-1, 2:, :] - pcd[1:-1, :-2, :]
    dy = pcd[2:, 1:-1, :] - pcd[:-2, 1:-1, :]
    # Cross product
    normal = np.cross(dx, dy)
    normal = normal / (np.linalg.norm(normal, axis=2, keepdims=True) + 1e-16)
    normal[depth == 0] = 0

    plt.imsave("normal.png", (normal+1)/2)

    pcd = pcd[1:-1, 1:-1, :]
    dx = DX(pcd)
    dy = DY(pcd)

    dxx= DX(dx)
    print(dxx[300,300])
    dyy= DY(dy)
    dxy= DX(dy)
    dyx= DY(dx)
    print(dxx.max(), dxx.min())
    diff = np.sum(dxy*dxy, axis=2) - np.sum(dyx*dyx, axis=2)
    diff = diff[100:-100, 100:-100]
    print(diff.max(), diff.min())

    plt.imsave("dxx.png", np.sum(dxx * dxx, axis=2)[100:-100, 100:-100])
    plt.imsave("dyy.png", np.sum(dyy * dyy, axis=2)[100:-100, 100:-100])
    plt.imsave("dxy.png", np.sum(dxy * dxy, axis=2)[100:-100, 100:-100])
    plt.imsave("dyx.png", np.sum(dyx * dyx, axis=2)[100:-100, 100:-100])

get_L(pcd)