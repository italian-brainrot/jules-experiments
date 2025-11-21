import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load the trajectories
trajectories = np.load('cg_curve_prediction/cg_trajectories.npy')

# Define the exponential decay function
def exp_decay(t, a, b, c):
    return a * np.exp(-b * t) + c

# Fit the model to each parameter trajectory
for i in range(trajectories.shape[1]):
    t_data = np.arange(trajectories.shape[0])
    p_data = trajectories[:, i]

    # Fit the curve
    popt, pcov = curve_fit(exp_decay, t_data, p_data, p0=(1, 1e-2, 0), maxfev=5000)

    print(f"Parameter {i+1}: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")

    # Plot the original and fitted curves for comparison
    plt.figure(figsize=(8, 6))
    plt.plot(t_data, p_data, label='Original')
    plt.plot(t_data, exp_decay(t_data, *popt), label='Fitted')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title(f'Parameter {i+1} Trajectory and Fitted Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'cg_curve_prediction/param_{i+1}_fit.png')

print("Saved trajectory plots to 'cg_curve_prediction/param_i_fit.png' files.")
