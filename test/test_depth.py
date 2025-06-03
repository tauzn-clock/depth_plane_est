import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from process_depth import get_normal

import numpy as np
from PIL import Image

img = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"rgb.png"))
img = np.array(img)

depth = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"depth.png"))
depth = np.array(depth)/ 1000.0

intrinsic = [306.9346923828125,
             306.8908386230469,
             318.58868408203125,
             198.37969970703125,
            ]

normal = get_normal(depth, intrinsic)

import matplotlib.pyplot as plt

plt.imsave("normal.png", (normal + 1) / 2.0)