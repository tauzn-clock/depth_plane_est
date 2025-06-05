import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"rgb.png"))
img = np.array(img)

depth = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"depth.png"))
depth = np.array(depth)/ 1000.0

intrinsic = [306.9346923828125,
             306.8908386230469,
             318.58868408203125,
             198.37969970703125,
            ]

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))

from process_depth import get_normal
normal = get_normal(depth, intrinsic)

plt.imsave("normal.png", (normal + 1) / 2.0)

from get_planes import get_planes
mask, param = get_planes(depth, intrinsic, 3, 0.02)

from mask_to_hsv import mask_to_hsv
plt.imsave("mask.png", mask_to_hsv(mask))

from save_pcd import save_pcd
from process_depth import get_3d
save_pcd(img, get_3d(depth, intrinsic), "output.ply")

from save_pcd import save_planes
save_planes(get_3d(depth, intrinsic), mask, param, "planes.ply")