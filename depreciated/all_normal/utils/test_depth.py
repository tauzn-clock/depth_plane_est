import numpy as np

def parabolic(H,W):
    depth = np.zeros((W, H), dtype = np.float32)

    for i in range(W):
        for j in range(H):
            depth[i,j] = 0.05 * ((i - W/2)**2 + (j - H/2)**2)
            
    depth = depth / np.max(depth) * 10

    return depth

def flat(H,W):
    depth = np.ones((W, H), dtype = np.float32)
    depth = depth / np.max(depth) * 5
    return depth