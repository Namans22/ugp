import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_normal_curve(mean, median, std_dev):
    # Generate normal distribution data (full range, no clipping)
    x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 1000)
    y = norm.pdf(x, mean, std_dev)  # Probability density function
    
    # Plot the normal curve
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='Normal Distribution', linewidth=2)
    
    # Mark mean, median, and std deviations
    plt.axvline(x=mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.axvline(x=median, color='g', linestyle='--', label=f'Median: {median:.2f}')
    
    # Mark ±1 and ±2 standard deviations
    plt.axvline(x=mean + std_dev, color='m', linestyle=':', label=f'±1 Std Dev: {std_dev:.2f}')
    plt.axvline(x=mean - std_dev, color='m', linestyle=':')
    plt.axvline(x=mean + 2*std_dev, color='k', linestyle=':', alpha=0.5, label='±2 Std Dev')
    plt.axvline(x=mean - 2*std_dev, color='k', linestyle=':', alpha=0.5)
    
    # Highlight the peak (mean)
    plt.scatter(mean, norm.pdf(mean, mean, std_dev), color='r', s=100, zorder=5)
    
    # Set x-axis limits to [0, 89] (no clipping, just focus)
    plt.xlim(0, 89)
    
    # Annotations
    plt.title("Normal Distribution Curve (Data Spread: 0 to 89)")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Example usage:
mean = 63.71
median = 66.12
std_dev = 17.29

plot_normal_curve(mean, median, std_dev)