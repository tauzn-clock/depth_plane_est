from scipy.spatial.transform import Rotation as R
import numpy as np

def gravity_correction(grav_normal, img_normal, pts_3d, bound, iteration):
    for _ in range(iteration):
        normal_x = np.array([0,grav_normal[1], grav_normal[2]])
        normal_x = normal_x / (np.linalg.norm(normal_x) + 1e-15)
        img_normal_x = img_normal.copy()
        img_normal_x[:, 0] = 0
        img_normal_x = img_normal_x / (np.linalg.norm(img_normal_x, axis=1, keepdims=True) + 1e-15)

        cos_x = img_normal_x[:,1] * normal_x[1] + img_normal_x[:,2] * normal_x[2]
        sin_x = img_normal_x[:,2] * normal_x[1] - img_normal_x[:,1] * normal_x[2]
        x = np.arctan2(sin_x, cos_x)

        normal_y = np.array([grav_normal[0], 0, grav_normal[2]])
        normal_y = normal_y / (np.linalg.norm(normal_y) + 1e-15)
        img_normal_y = img_normal.copy()
        img_normal_y[:, 1] = 0
        img_normal_y = img_normal_y / (np.linalg.norm(img_normal_y, axis=1, keepdims=True) + 1e-15)

        cos_y = img_normal_y[:,0] * normal_y[0] + img_normal_y[:,2] * normal_y[2]
        sin_y = -img_normal_y[:,2] * normal_y[0] +img_normal_y[:,0] * normal_y[2]
        y = np.arctan2(sin_y, cos_y)

        angle_dist = np.dot(img_normal, grav_normal)
        mean_x = np.mean(x[(angle_dist > bound) & (pts_3d[:,2]!=0)])
        mean_y = np.mean(y[(angle_dist > bound) & (pts_3d[:,2]!=0)])

        rot = R.from_euler("XYZ",[mean_x,mean_y,0]).as_matrix()
        grav_normal = rot @ grav_normal

        #normal = np.array([grav_normal[0], np.cos(mean_x) * grav_normal[1] - np.sin(mean_x) * grav_normal[2], np.sin(mean_x) * grav_normal[1] + np.cos(mean_x) * grav_normal[2]])
        #normal = np.array([np.cos(mean_y) * grav_normal[0] + np.sin(mean_y) * grav_normal[2], grav_normal[1], -np.sin(mean_y) * grav_normal[0] + np.cos(mean_y) * grav_normal[2]])
    
    return grav_normal