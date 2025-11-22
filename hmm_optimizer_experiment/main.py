import numpy as np
from .optimizer import hmm_minimize
from .test_functions import rosenbrock, ackley
from .visualize import plot_contour

def run_experiment(func, bounds, func_name):
    """
    Runs the HMM-based optimization for a given function and visualizes the result.
    """
    best_x, best_val, path = hmm_minimize(func, bounds, n_iter=100, n_particles=50, random_state=np.random.RandomState(0))

    print(f"Best solution for {func_name}: x = {best_x}, f(x) = {best_val}")

    plot_contour(func, bounds, path, f"{func_name} Optimization with HMM")

if __name__ == "__main__":
    rosenbrock_bounds = [(-2, 2), (-2, 2)]
    run_experiment(rosenbrock, rosenbrock_bounds, "Rosenbrock")

    ackley_bounds = [(-5, 5), (-5, 5)]
    run_experiment(ackley, ackley_bounds, "Ackley")
