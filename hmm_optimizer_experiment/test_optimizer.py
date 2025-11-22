import numpy as np
from .optimizer import hmm_minimize
from .test_functions import rosenbrock

def test_hmm_minimize():
    """
    Tests the hmm_minimize function on the Rosenbrock function.
    """
    bounds = [(-2, 2), (-2, 2)]
    best_x, best_val, _ = hmm_minimize(rosenbrock, bounds, n_iter=100, n_particles=50, random_state=np.random.RandomState(0))

    # Check that the result is within a reasonable range of the known minimum
    assert np.allclose(best_x, [1, 1], atol=0.5)
    assert best_val < 1.0
