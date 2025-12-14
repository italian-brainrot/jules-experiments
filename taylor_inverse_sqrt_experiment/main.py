import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import svd
import os

def inv_sqrt_eig(A):
    """Computes the inverse square root of a symmetric matrix using eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    inv_sqrt_eigenvalues = 1.0 / np.sqrt(eigenvalues)
    return eigenvectors @ np.diag(inv_sqrt_eigenvalues) @ eigenvectors.T

def inv_sqrt_svd(A):
    """Computes the inverse square root of a symmetric matrix using SVD."""
    U, s, Vh = svd(A)
    return U @ np.diag(1.0 / np.sqrt(s)) @ Vh

def inv_sqrt_taylor(A, num_terms=20):
    """
    Computes the inverse square root of a symmetric matrix using a Taylor series expansion.
    """
    I = np.eye(A.shape[0])

    vals = np.linalg.eigvalsh(A)
    lambda_min, lambda_max = vals[0], vals[-1]

    if np.allclose(lambda_min, 1.0) and np.allclose(lambda_max, 1.0):
        return I

    c = (lambda_max + lambda_min) / 2
    d = (lambda_max - lambda_min) / 2

    if np.isclose(d, 0):
        return I / np.sqrt(c)

    B = (A - c * I) / d
    B_prime = -(d/c) * B

    inv_sqrt_B_prime = I
    term = I
    for i in range(1, num_terms):
        term = term @ B_prime * (2 * i - 1) / (2 * i)
        inv_sqrt_B_prime += term

    return inv_sqrt_B_prime / np.sqrt(c)


def generate_spd_matrix(n):
    """Generates a symmetric positive-definite matrix of size n x n."""
    A = np.random.rand(n, n)
    return A @ A.T + np.eye(n) * 1e-3

def benchmark():
    """Benchmarks the implemented algorithms."""
    matrix_sizes = [10, 20, 50, 100, 200, 500]
    num_terms_list = [5, 10, 20, 40]

    results = {
        'eig': {'times': [], 'errors': []},
        'svd': {'times': [], 'errors': []},
    }
    for num_terms in num_terms_list:
        results[f'taylor_{num_terms}'] = {'times': [], 'errors': []}

    for n in matrix_sizes:
        print(f"Benchmarking for matrix size: {n}x{n}")
        A = generate_spd_matrix(n)
        A_inv = np.linalg.inv(A)

        # Benchmark eig
        start_time = time.time()
        inv_sqrt_A_eig = inv_sqrt_eig(A)
        end_time = time.time()
        results['eig']['times'].append(end_time - start_time)
        results['eig']['errors'].append(np.linalg.norm(inv_sqrt_A_eig @ inv_sqrt_A_eig - A_inv))

        # Benchmark svd
        start_time = time.time()
        inv_sqrt_A_svd = inv_sqrt_svd(A)
        end_time = time.time()
        results['svd']['times'].append(end_time - start_time)
        results['svd']['errors'].append(np.linalg.norm(inv_sqrt_A_svd @ inv_sqrt_A_svd - A_inv))

        # Benchmark taylor
        for num_terms in num_terms_list:
            start_time = time.time()
            inv_sqrt_A_taylor = inv_sqrt_taylor(A, num_terms=num_terms)
            end_time = time.time()
            results[f'taylor_{num_terms}']['times'].append(end_time - start_time)
            results[f'taylor_{num_terms}']['errors'].append(np.linalg.norm(inv_sqrt_A_taylor @ inv_sqrt_A_taylor - A_inv))

    return matrix_sizes, results

def plot_results(matrix_sizes, results):
    """Plots the benchmarking results."""

    # Create a directory to save the plots
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Plot execution times
    plt.figure(figsize=(10, 6))
    for method, data in results.items():
        plt.plot(matrix_sizes, data['times'], marker='o', label=method)
    plt.xlabel("Matrix Size (n)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time of Matrix Inverse Square Root Methods")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(script_dir, "execution_times.png"))

    # Plot reconstruction errors (log scale)
    plt.figure(figsize=(10, 6))
    for method, data in results.items():
        plt.plot(matrix_sizes, data['errors'], marker='o', label=method)
    plt.xlabel("Matrix Size (n)")
    plt.ylabel("Reconstruction Error (Frobenius Norm)")
    plt.title("Reconstruction Error of Matrix Inverse Square Root Methods")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(script_dir, "reconstruction_errors.png"))

if __name__ == '__main__':
    matrix_sizes, results = benchmark()
    plot_results(matrix_sizes, results)
