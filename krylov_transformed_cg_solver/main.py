
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator, cg
import matplotlib.pyplot as plt
import os

def lanczos_iteration(A_op, v, m):
    """
    Performs m steps of the Lanczos iteration to build a Krylov subspace.

    Args:
        A_op (LinearOperator): The linear operator for the matrix A.
        v (np.ndarray): The starting vector.
        m (int): The dimension of the Krylov subspace.

    Returns:
        Q (np.ndarray): An (n x m) orthonormal basis for the Krylov subspace.
        T (np.ndarray): An (m x m) tridiagonal matrix.
    """
    n = len(v)
    Q = np.zeros((n, m))
    alphas = np.zeros(m)
    betas = np.zeros(m - 1)

    q = v / np.linalg.norm(v)
    Q[:, 0] = q

    for j in range(m):
        w = A_op @ Q[:, j]
        alphas[j] = np.dot(w, Q[:, j])

        if j == 0:
            w = w - alphas[j] * Q[:, j]
        else:
            w = w - alphas[j] * Q[:, j] - betas[j-1] * Q[:, j-1]

        if j < m - 1:
            beta = np.linalg.norm(w)
            if beta < 1e-10: # Breakdown
                T = np.diag(alphas[:j+1]) + np.diag(betas[:j], 1) + np.diag(betas[:j], -1)
                return Q[:, :j+1], T
            betas[j] = beta
            Q[:, j+1] = w / beta

    T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
    return Q, T

def kt_cg_solver(A_op, b, x_true, m, max_iter=150):
    """
    Solves Ax = A^(1/2)b by first approximating A^(1/2)b in a Krylov subspace.
    """
    Q, T = lanczos_iteration(A_op, b, m)
    T_sqrt = scipy.linalg.sqrtm(T)

    b_krylov = np.zeros(m)
    b_krylov[0] = np.linalg.norm(b)

    b_hat = Q @ T_sqrt @ b_krylov

    errors = []
    matvec_counts = []
    total_matvecs = m

    def callback(xk):
        nonlocal total_matvecs
        total_matvecs += 1
        error = np.linalg.norm(xk - x_true)
        errors.append(error)
        matvec_counts.append(total_matvecs)

    x, _ = cg(A_op, b_hat, maxiter=max_iter, callback=callback, rtol=1e-12)

    return x, errors, matvec_counts

def baseline_solver(A_op, A_sqrt_b, x_true, max_iter=150):
    """
    Solves the transformed system Ax = A^(1/2)b using standard CG.
    """
    errors = []
    matvec_counts = []

    def callback(xk):
        count = len(matvec_counts) + 1
        error = np.linalg.norm(xk - x_true)
        errors.append(error)
        matvec_counts.append(count)

    x, _ = cg(A_op, A_sqrt_b, maxiter=max_iter, callback=callback, rtol=1e-12)
    return x, errors, matvec_counts

def main():
    n = 256
    np.random.seed(0)
    A_rand = np.random.rand(n, n)
    A_sym = (A_rand + A_rand.T) / 2
    A = A_sym + n * np.eye(n)

    A_op = LinearOperator((n, n), matvec=lambda v: A @ v)

    A_sqrt = scipy.linalg.sqrtm(A)
    x_true = np.random.rand(n)

    # This is the b for the original system A^(1/2)x = b
    b_orig = A_sqrt @ x_true

    # The baseline solves the transformed system Ax = A^(1/2)b_orig
    A_sqrt_b_orig = A @ x_true

    max_cg_iter = n

    _, baseline_errors, baseline_matvecs = baseline_solver(A_op, A_sqrt_b_orig, x_true, max_iter=max_cg_iter)

    subspace_dims = [5, 15, 30]
    results = {}
    for m in subspace_dims:
        _, errors, matvecs = kt_cg_solver(A_op, b_orig, x_true, m=m, max_iter=max_cg_iter)
        results[m] = (errors, matvecs)

    plt.figure(figsize=(10, 6))
    plt.plot(baseline_matvecs, baseline_errors, label='Baseline (Standard CG)', color='black', linestyle='--')

    for m in subspace_dims:
        errors, matvecs = results[m]
        plt.plot(matvecs, errors, label=f'KT-CG (m={m})')

    plt.yscale('log')
    plt.xlabel('Number of Matrix-Vector Products with A')
    plt.ylabel('Error Norm ||x_k - x_true||')
    plt.title('Krylov-Transformed CG vs. Standard CG')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'convergence_comparison.png'))
    print("Plot saved to convergence_comparison.png")

if __name__ == '__main__':
    main()
