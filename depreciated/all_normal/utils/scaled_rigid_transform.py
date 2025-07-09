import numpy as np

def scaled_rigid_transform(x,y):
    """
    Scaled Iterative Closest Point (ICP) algorithm to align two point clouds.
    
    Parameters:
        x (np.ndarray): Source point cloud of shape (N, 3).
        y (np.ndarray): Target point cloud of shape (M, 3).
        
    Returns:
        np.ndarray: Transformed source point cloud.
    """
    # Compute centroids
    centroid_x = np.mean(x, axis=0)
    centroid_y = np.mean(y, axis=0)

    # Center the point clouds
    x_centered = x - centroid_x
    y_centered = y - centroid_y

    # Compute covariance matrix
    H = np.dot(x_centered.T, y_centered)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = np.dot(Vt.T, U.T)

    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute scaling factor
    s = np.sum(S) / np.sum(x_centered**2)

    # Compute translation vector
    t = centroid_y - s * np.dot(R, centroid_x)

    # Apply transformation
    x_transformed = s * np.dot(x, R.T) + t

    return s, R, t

if __name__ == "__main__":
    # Example usage
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    #y = np.array([[0, 1, 1], [0, 0, 2], [0, 0, 0]])
    R = np.array([[1, 0, 0], [0, 0.707, 0.707], [0, -0.707, 0.707]])
    s = 2
    y = s * np.dot(x, R.T) + np.array([1, 1, 1])

    transformed_x = scaled_rigid_transform(x, y)
    print("Transformed Source Point Cloud:\n", transformed_x)