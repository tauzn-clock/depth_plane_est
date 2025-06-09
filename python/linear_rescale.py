import numpy as np
import matplotlib.pyplot as plt

def linear_rescale(depth, est):

    # Find depth = m * est + b
    
    # We want to find m and b such that the error is minimized
    # We can use np.linalg.lstsq to find the best fit line
    
    A = np.vstack([est, np.ones_like(est)]).T
    b = depth
    
    m, b = np.linalg.lstsq(A, b, rcond=None)[0]
    
    return m, b

def plot_rescale(depth, est, m, b, path):
    fig, ax = plt.subplots()
    
    ax.scatter(est, depth, s=1, alpha=0.5, label='Data Points')
    # Plot the best fit line
    x_fit = np.linspace(est.min(), est.max(), 100)
    y_fit = m * x_fit + b
    ax.plot(x_fit, y_fit, color='red', label='Best Fit Line')
    
    ax.set_xlabel('Estimated Depth (m)')
    ax.set_ylabel('Ground Truth Depth (m)')
    
    plt.savefig(path)

def get_metrics(depth, est, m, b, path):
    est = m * est + b
    
    diff = np.abs(depth - est)

    # Save histogram
    fig, ax = plt.subplots()
    ax.hist(diff, bins=100)
    ax.set_xlabel("Absolute error (m)")
    ax.set_ylabel("Frequency")

    plt.savefig(path)

    threshold = np.maximum((depth / est), (est / depth))
    delta1 = (threshold < 1.25).mean()
    print(f"Delta1: {delta1:.4f}")
    
    rmse = np.sqrt(np.mean((depth - est) ** 2))
    print(f"RMSE: {rmse:.4f} m")
    
    percentile_95 = np.percentile(diff, 95)
    print(f"95th Percentile Error: {percentile_95:.4f} m")
    
