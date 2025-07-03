import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import torch
import json
import os
from PIL import Image
import matplotlib.pyplot as plt

from utils.metric3d import metric3d

DIR = "/scratchdata/processed/stair4"
with open(os.path.join(DIR, "camera_info.json"), "r") as f:
    camera_info = json.load(f)
INTRINSICS = camera_info["P"]
print(INTRINSICS)

model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True).cuda() 
model = model.cuda() if torch.cuda.is_available() else model