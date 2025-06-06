import numpy as np
from .process_depth import get_3d
from .orientation_correction import orientation_correction

def get_planes(depth, normal, INTRINSICS, ANGLE_CLUSTER, RATIO_SIZE):
    H, W = depth.shape
    points = get_3d(depth, INTRINSICS)
        
    ANGLE_INCREMENT = 37 # Divide into 5 deg bins
    KERNEL_2D = 5 # Extract max in 20 deg range
    
    img_normal_angle = np.zeros((H, W, 2))
    img_normal_angle[:, :, 0] = np.arctan(normal[:,:,0]/(normal[:,:,2]+1e-15))
    img_normal_angle[:, :, 1] = np.arctan(normal[:,:,1]/(normal[:,:,2]+1e-15))

    angle_cluster = np.zeros((ANGLE_INCREMENT, ANGLE_INCREMENT))
    for i in range(0, ANGLE_INCREMENT):
        for j in range(0, ANGLE_INCREMENT):
            angle_cluster[i, j] = np.sum((img_normal_angle[:, :, 0] >= (i-ANGLE_INCREMENT/2)*np.pi/ANGLE_INCREMENT ) 
                                         & (img_normal_angle[:, :, 0] < (i+1-ANGLE_INCREMENT/2)*np.pi/ANGLE_INCREMENT)
                                         & (img_normal_angle[:, :, 1] >= (j-ANGLE_INCREMENT/2)*np.pi/ANGLE_INCREMENT)
                                         & (img_normal_angle[:, :, 1] < (j+1-ANGLE_INCREMENT/2)*np.pi/ANGLE_INCREMENT)
                                         & (depth != 0))

    dillation = np.zeros((ANGLE_INCREMENT, ANGLE_INCREMENT))
    angle_cluster = np.pad(angle_cluster, ((KERNEL_2D//2, KERNEL_2D//2), (KERNEL_2D//2, KERNEL_2D//2)), mode='wrap')
    for i in range(ANGLE_INCREMENT):
        for j in range(ANGLE_INCREMENT):
            dillation[i, j] = np.max(angle_cluster[i:i+KERNEL_2D, j:j+KERNEL_2D])
    angle_cluster = angle_cluster[KERNEL_2D//2:-KERNEL_2D//2+1, KERNEL_2D//2:-KERNEL_2D//2+1]
    
    # Find index where angle_cluster is equal to dillation
    index = np.where(angle_cluster == dillation)

    # Get the best peaks in the angle_cluster
    best_peaks = np.argsort(angle_cluster[index])[-ANGLE_CLUSTER:][::-1]
    best_peaks_index = np.array([index[0][best_peaks], index[1][best_peaks]]).T

    mask = np.zeros_like(depth, dtype=np.int8)
    param = np.array([])

    for i in range(best_peaks_index.shape[0]):
        angle_x = (best_peaks_index[i][0] - ANGLE_INCREMENT//2) * np.pi / ANGLE_INCREMENT
        angle_y = (best_peaks_index[i][1] - ANGLE_INCREMENT//2) * np.pi / ANGLE_INCREMENT
        
        grav_normal = np.array([np.tan(angle_x), np.tan(angle_y), 1])
        grav_normal = grav_normal / np.linalg.norm(grav_normal)
        
        dot_bound = 0.9
        correction_iteration = 5
        kernel_size = 11
        cluster_size = 5
        
        grav_normal = orientation_correction(grav_normal, normal.reshape(-1,3), dot_bound, correction_iteration)
        
        normal_abs = normal.reshape(-1, 3)
        normal_abs = normal_abs * np.where(np.dot(normal_abs, grav_normal) >= 0, 1, -1).reshape(-1, 1)

        new_mask, new_param = get_mask(grav_normal, normal_abs.reshape(-1,3), points.reshape(-1,3), dot_bound, kernel_size, cluster_size, RATIO_SIZE=RATIO_SIZE)
        new_mask = new_mask.reshape(H, W)
        mask = np.where((new_mask != 0), new_mask + mask.max(), mask)
        if new_param.size:
            param = np.concatenate((param, new_param), axis=0) if param.size else new_param
        
    return mask, param

def get_mask(grav_normal, img_normal, pts_3d, dot_bound, kernel_size, cluster_size, bin_size=0.01, RATIO_SIZE=0.1):
    mask = np.zeros(len(img_normal), dtype=np.uint8)
    param = []
    index = np.array([i for i in range(len(img_normal))])

    #Is this required?
    img_normal = img_normal * np.where(np.linalg.norm(img_normal, axis=1) > 0, 1, -1).reshape(-1, 1)

    angle_dist = np.dot(img_normal, grav_normal)
    scalar_dist = np.dot(pts_3d, grav_normal)
    scalar_dist[angle_dist < dot_bound] = 0
    scalar_dist[pts_3d[:, 2] == 0] = 0

    index = index[scalar_dist != 0]
    scalar_dist = scalar_dist[scalar_dist != 0]
    if (len(scalar_dist)==0):
        return mask
        
    bins = np.arange(scalar_dist.min(), scalar_dist.max(), bin_size)
    hist, bin_edges = np.histogram(scalar_dist, bins=bins)

    kernel = [-kernel_size//2 + 1 + i for i in range(kernel_size)]
    group = [-cluster_size//2 + 1 + i for i in range(cluster_size)]

    # Dilation of histogram
    dilation_hist = np.pad(hist, (kernel_size//2, kernel_size//2))
    dilation_hist = [np.roll(dilation_hist, i) for i in kernel]
    dilation_hist = np.array(dilation_hist)
    dilation_hist = np.max(dilation_hist, axis=0)[kernel_size//2:-kernel_size//2+1]
    
    # Get index where dilation_hist is equal to hist
    candidate_peak = np.where(dilation_hist == hist)[0]

    local_total = np.pad(hist, (cluster_size//2, cluster_size//2))
    local_total = [np.roll(local_total, i) for i in group]
    local_total = np.array(local_total)
    local_total = np.sum(local_total, axis=0)[cluster_size//2:-cluster_size//2+1]

    # Get index of the plane_cnt largest values in local_total
    best_peaks = np.argsort(local_total[candidate_peak])[::-1]
    best_peaks_index = candidate_peak[best_peaks]
    
    for i in range(len(best_peaks_index)):
        if (best_peaks_index[i] - cluster_size//2) >=0:
                lower_bound = bin_edges[best_peaks_index[i]-cluster_size//2]
        else:
            lower_bound = bin_edges[0]
        if (best_peaks_index[i] + 1 + cluster_size//2) < len(bin_edges):
            upper_bound = bin_edges[best_peaks_index[i] + 1 + cluster_size//2]
        else:
            upper_bound = bin_edges[-1]+ bin_size
            
        tmp_mask = np.zeros_like(mask, dtype=np.uint8)    
        for j in range(len(scalar_dist)):
            if scalar_dist[j] > lower_bound and scalar_dist[j] < upper_bound:
                tmp_mask[index[j]] = i + 1
        
        if np.sum(tmp_mask)/ len(tmp_mask) > RATIO_SIZE:
            mask = np.where(tmp_mask != 0, tmp_mask, mask)
            param.append(np.array([grav_normal[0], grav_normal[1], grav_normal[2], bin_edges[best_peaks_index[i]]]))
            #mask = np.where(tmp_mask != 0, mask.max()+1, mask)
            #masked_pts = pts_3d[tmp_mask.flatten() != 0]
            #param.append(estimate_plane_param_svd(masked_pts))
        else:
            break
        
    print(mask.max(), len(param))

    return mask, np.array(param)

def estimate_plane_param_svd(pcd):
    """
    Estimate plane parameters using Singular Value Decomposition (SVD)
    :param pcd: Point cloud data
    :return: Plane parameters [a, b, c, d] where ax + by + cz + d = 0
    """
    # Compute the centroid of the point cloud
    centroid = np.mean(pcd, axis=0)

    # Center the point cloud
    centered_pcd = pcd - centroid

    # Perform SVD
    covariance_matrix = np.dot(centered_pcd.T, centered_pcd)
    _, _, vh = np.linalg.svd(covariance_matrix)

    # The normal vector is the last row of vh
    normal = vh[-1]

    # Calculate d using the plane equation ax + by + cz + d = 0
    d = np.dot(normal, centroid)


    return np.append(normal, d)