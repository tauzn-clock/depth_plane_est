import numpy as np
import matplotlib.pyplot as plt

def inverse_linear_rescale(depth, est):
    
    est_flatten = est.flatten()
    depth = depth.flatten()
    
    est_flatten = est_flatten[depth > 0]  # Filter out invalid depth values
    depth = depth[depth > 0]  # Filter out invalid depth values
    
    # Find 1/depth = m * 1/est + b
    
    # We want to find m and b such that the error is minimized
    # We can use np.linalg.lstsq to find the best fit line
    
    A = np.vstack([1/(est_flatten.flatten()), np.ones_like(1/(est_flatten.flatten()))]).T
    b = 1/depth.flatten()
    
    m, b = np.linalg.lstsq(A, b, rcond=None)[0]
    
    print(f"Linear rescale parameters: m = {m:.4f}, b = {b:.4f}")
    
    fig, ax = plt.subplots()
    
    ax.scatter(est_flatten, depth, s=1, alpha=0.5, label='Data Points')
    # Plot the best fit line
    x_fit = np.linspace(est.min(), est.max(), 100)
    y_fit = m * (1/x_fit) + b
    ax.plot(x_fit, 1/y_fit, color='red', label='Best Fit Line')
    
    ax.set_xlabel('Estimated Depth (m)')
    ax.set_ylabel('Ground Truth Depth (m)')
    
    plt.savefig("inverse_linear_rescale.png")
    
    return 1/(m*1 / est + b)