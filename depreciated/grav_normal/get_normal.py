from depth_to_pcd import depth_to_pcd
import numpy as np

def get_normal(depth, INTRINSICS):
    H, W = depth.shape
    points, _ = depth_to_pcd(depth.flatten(), INTRINSICS, W, H)

    points = points.reshape(H, W, 3)

    #Pad points along the edges
    points = np.pad(points, ((1, 1), (1, 1), (0, 0)))

    dx = points[1:-1, 2:, :] - points[1:-1, :-2, :]
    dy = points[2:, 1:-1, :] - points[:-2, 1:-1, :]
    # Cross product
    normal = np.cross(dx, dy)
    normal = normal / (np.linalg.norm(normal, axis=2, keepdims=True) + 1e-16)
    normal[depth == 0] = 0

    return normal
    