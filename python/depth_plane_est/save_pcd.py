import open3d as o3d
import numpy as np
from .mask_to_hsv import mask_to_hsv
from .projected_convex_hull import projected_convex_hull

def save_pcd(img, pcd, path): 
    img = img.reshape(-1, 3)
    pcd = pcd.reshape(-1, 3)

    img = img[pcd[:,2]!=0]
    pcd = pcd[pcd[:,2]!=0]

    output = o3d.geometry.PointCloud()
    output.points = o3d.utility.Vector3dVector(pcd)
    output.colors = o3d.utility.Vector3dVector(img.astype(np.float64)/255.0)

    o3d.io.write_point_cloud(path, output)
    
    return output    

def get_convex_hull_mesh(pcd, color):
    avg = np.mean(pcd, axis=0)
    
    # Open3d TriangleMesh expects points in a specific format
    mesh = o3d.geometry.TriangleMesh()
        
    triangles = []
    
    for i in range(len(pcd)):
        triangles.append([i, (i+1) % len(pcd), len(pcd)])
    
    pcd = np.vstack((pcd, avg.reshape(1, 3)))

    mesh.vertices = o3d.utility.Vector3dVector(pcd)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color(color / 255.0)
    mesh.compute_vertex_normals()
    
    return mesh
    
   
def save_planes(pcd, mask, param, path):
    color = mask_to_hsv(mask)
    
    mesh = o3d.geometry.TriangleMesh()
    
    for i in range(mask.max()):
        plane_pcd = pcd[mask.flatten() == (i+1)]
        plane_color = color[mask == (i+1)]
        plane_param = param[i]
        
        convex_hull = projected_convex_hull(plane_pcd, plane_param)
        
        mesh += get_convex_hull_mesh(convex_hull, plane_color[0])
    
    o3d.io.write_triangle_mesh(path, mesh)
    
    return mesh