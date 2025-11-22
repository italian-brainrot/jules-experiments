import numpy as np

def rosenbrock(x):
    """The Rosenbrock function"""
    x = np.asarray(x)
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0, axis=0)

def ackley(x):
    """The Ackley function"""
    x = np.asarray(x)
    n = x.shape[0]
    sum_sq_term = -0.2 * np.sqrt(np.sum(x**2, axis=0) / n)
    cos_term = np.sum(np.cos(2.0 * np.pi * x), axis=0) / n
    return -20.0 * np.exp(sum_sq_term) - np.exp(cos_term) + 20.0 + np.e
