#from scipy.spatial.transform import Rotation as R
import numpy as np

def orientation_correction(grav_normal, img_normal, bound, iteration):
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
        mean_x = np.mean(x[(angle_dist > bound) & (img_normal[:,2]!=0)])
        mean_y = np.mean(y[(angle_dist > bound) & (img_normal[:,2]!=0)])

        #rot = R.from_euler("XYZ",[mean_x,mean_y,0]).as_matrix()
        rot = get_rotation(mean_x, mean_y, 0)
        grav_normal = rot @ grav_normal
        
    return grav_normal

def get_rotation(alpha, beta, gamma):
    """
    Get rotation matrix from Euler angles
    :param alpha: rotation around x-axis
    :param beta: rotation around y-axis
    :param gamma: rotation around z-axis
    :return: rotation matrix
    """
    R = np.array([[np.cos(beta) * np.cos(gamma), -np.cos(beta) * np.sin(gamma), np.sin(beta)],
                 [np.cos(alpha) * np.sin(gamma) + np.cos(gamma) * np.sin(alpha) * np.sin(beta), np.cos(alpha) * np.cos(gamma) - np.sin(alpha) * np.sin(beta) * np.sin(gamma), -np.cos(beta) * np.sin(alpha)],
                 [np.sin(alpha) * np.sin(gamma) - np.cos(alpha) * np.cos(gamma) * np.sin(beta), np.cos(gamma) * np.sin(alpha) + np.cos(alpha) * np.sin(beta) * np.sin(gamma), np.cos(alpha) * np.cos(beta)]])

    return R