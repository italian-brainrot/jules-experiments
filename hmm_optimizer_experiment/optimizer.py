import numpy as np
from hmmlearn import hmm

def hmm_minimize(func, bounds, n_iter=1000, n_particles=50, n_hidden_states=4, random_state=None):
    """
    Minimize a function using a Hidden Markov Model-based optimization algorithm.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # Initial random search
    points = random_state.uniform(lb, ub, size=(n_particles, dim))
    values = np.array([func(p) for p in points])

    path = []
    # Main optimization loop
    for _ in range(n_iter):
        # Sort points by function value
        sorted_indices = np.argsort(values)
        points = points[sorted_indices]
        values = values[sorted_indices]

        path.append(np.copy(points[0]))

        # Fit HMM to the sorted points
        model = hmm.GaussianHMM(n_components=n_hidden_states, covariance_type="diag", n_iter=100, random_state=random_state)
        model.fit(points)

        # Sample new points from the HMM
        new_points, _ = model.sample(n_particles)

        # Evaluate new points
        new_values = np.array([func(p) for p in new_points])

        # Replace worst half of points with new points
        n_replace = n_particles // 2
        points[-n_replace:] = new_points[:n_replace]
        values[-n_replace:] = new_values[:n_replace]

    best_idx = np.argmin(values)
    return points[best_idx], values[best_idx], path
