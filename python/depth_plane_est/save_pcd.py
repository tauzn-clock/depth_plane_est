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
    
    for i in range(len(param)):
        plane_pcd = pcd[mask.flatten() == (i+1)]
        plane_color = color[mask == (i+1)]
        plane_param = param[i]
        
        convex_hull = projected_convex_hull(plane_pcd, plane_param)
        
        mesh += get_convex_hull_mesh(convex_hull, plane_color[0])
    
    o3d.io.write_triangle_mesh(path, mesh)
    
    return mesh

def save_normal(pcd, normal, path):
    pcd = pcd.reshape(-1, 3)
    normal = normal.reshape(-1, 3)

    normal = normal[pcd[:, 2] != 0]
    pcd = pcd[pcd[:, 2] != 0]

    pcd = pcd[np.linalg.norm(normal, axis=1) > 0]
    normal = normal[np.linalg.norm(normal, axis=1) > 0]

    #Down sample
    pcd = pcd[::10]
    normal = normal[::10]

    mesh = o3d.geometry.TriangleMesh()
    
    for i in range(len(pcd)):
        start = pcd[i]
        direction = normal[i]
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.1, cone_height=0.05)
        tf = np.eye(4)
        tf[:3, :3] = rotation_matrix_from_vectors(np.array([0, 0, 1]), direction)
        tf[:3, 3] = start
        arrow.transform(tf)
        arrow.paint_uniform_color([1, 0, 0])  # Red color for the arrows
        mesh += arrow

    o3d.io.write_triangle_mesh(path, mesh)

    return mesh

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A 3x3 rotation matrix that aligns vec1 with vec2
    """
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    K = np.array([[0, -v[2], v[1]],
              [v[2], 0, -v[0]],
              [-v[1], v[0], 0]])

    R = np.eye(3) + s * K + (1 - c) * np.dot(K, K)

    if np.linalg.det(R) < 0:
        R = -R

    return R