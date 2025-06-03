import open3d as o3d
import numpy as np

def save_pcd(img, pcd, path): 
    img = img.reshape(-1, 3)
    pcd = pcd.reshape(-1, 3)

    img = img[pcd[:,2]!=0]
    pcd = pcd[pcd[:,2]!=0]

    output = o3d.geometry.PointCloud()
    output.points = o3d.utility.Vector3dVector(pcd)
    output.colors = o3d.utility.Vector3dVector(img.astype(np.float64)/255.0)

    o3d.io.write_point_cloud(path, output)