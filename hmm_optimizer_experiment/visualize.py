import numpy as np
import matplotlib.pyplot as plt

def plot_contour(func, bounds, path, title):
    """
    Plots the contour of a 2D function and the optimization path.
    """
    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = func([X, Y])

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Function Value')

    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'r-o', label='Optimization Path')

    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"hmm_optimizer_experiment/results/{title.replace(' ', '_').lower()}.png")
    plt.close()
