import numpy as np

def get_mask(grav_normal, img_normal, pts_3d, dot_bound, kernel_size, cluster_size, bin_size=0.01, RATIO_SIZE=0.1):
    mask = np.zeros(len(img_normal), dtype=np.uint8)
    index = np.array([i for i in range(len(img_normal))])

    dot1 = np.dot(img_normal, grav_normal).reshape(-1,1)
    dot2 = np.dot(img_normal, -grav_normal).reshape(-1,1)

    angle_dist = np.concatenate((dot1, dot2), axis=1)
    angle_dist = np.max(angle_dist, axis=1)
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
        else:
            break
    
    return mask