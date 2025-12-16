
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse.linalg import LinearOperator, cg

def lanczos_iteration(A_op, n_steps, v_start=None):
    """
    Performs the Lanczos iteration to generate a tridiagonal matrix T and an orthogonal basis Q.
    """
    n = A_op.shape[0]
    if v_start is None:
        q = np.random.rand(n)
        q /= np.linalg.norm(q)
    else:
        q = v_start / np.linalg.norm(v_start)

    Q = np.zeros((n, n_steps))
    alphas = []
    betas = []

    for j in range(n_steps):
        Q[:, j] = q
        v = A_op @ q
        alpha = np.dot(q, v)
        alphas.append(alpha)

        v = v - alpha * q
        if j > 0:
            v = v - betas[j-1] * Q[:, j-1]

        beta = np.linalg.norm(v)
        if beta > 1e-10 and j < n_steps - 1:
            betas.append(beta)
            q = v / beta
        else:
            break

    T = np.diag(alphas)
    if betas:
        T += np.diag(betas, k=1)
        T += np.diag(betas, k=-1)

    return T, Q[:, :len(alphas)]

def lanczos_sqrt_mv(A_op, b, n_steps):
    """
    Approximates A^{1/2}b using the Lanczos algorithm with eigendecomposition of T.
    """
    T, Q = lanczos_iteration(A_op, n_steps, v_start=b)

    eigvals, eigvecs = np.linalg.eigh(T)
    sqrt_T = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0))) @ eigvecs.T

    e1 = np.zeros(T.shape[0])
    e1[0] = 1.0
    b_hat = np.linalg.norm(b) * Q @ (sqrt_T @ e1)

    return b_hat

def lanczos_cg_solver(A_op, b, lanczos_steps):
    """
    Solves Ax = A^{1/2}b using Lanczos approximation and CG.
    """
    b_hat = lanczos_sqrt_mv(A_op, b, n_steps=lanczos_steps)
    x, info = cg(A_op, b_hat)
    return x

def estimate_eigenvalues(A_op, n_steps=30):
    """
    Estimates the extremal eigenvalues of a linear operator A using the Lanczos algorithm.
    """
    T, _ = lanczos_iteration(A_op, n_steps)
    eigvals = np.linalg.eigvalsh(T)
    return np.min(eigvals), np.max(eigvals)

def chebyshev_sqrt_solver(A_op, b, lambda_min, lambda_max, degree=20):
    # Map the interval [lambda_min, lambda_max] to [-1, 1]
    g = lambda y: (y * (lambda_max - lambda_min) + (lambda_max + lambda_min)) / 2
    f = lambda y: g(y)**(-0.5)

    # --- FIX: Use Chebyshev Nodes for fitting ---
    # Generate roots of the Chebyshev polynomial of degree 'num_points'
    # These cluster at the edges (-1 and 1) to prevent Runge's phenomenon.
    # num_points = degree + 1  # Or higher, e.g., 100
    # k = np.arange(1, num_points + 1)
    num_points = 2 * degree + 1
    k = np.arange(1, num_points + 1)
    cheb_nodes = np.cos((2 * k - 1) / (2 * num_points) * np.pi)

    # Fit on these nodes
    coeffs = np.polynomial.chebyshev.chebfit(cheb_nodes, f(cheb_nodes), degree)
    # ------------------------------------------

    # Clenshaw's algorithm (Forward Summation)
    # Note: Ensure lambda_max != lambda_min to avoid division by zero
    diff = lambda_max - lambda_min
    avg = (lambda_max + lambda_min) / 2

    w0 = b
    w1 = (A_op @ b - avg * b) * (2 / diff)

    res = coeffs[0] * w0 + coeffs[1] * w1

    for i in range(2, degree + 1):
        wi = 2 * ((A_op @ w1 - avg * w1) * (2 / diff)) - w0
        res = res + coeffs[i] * wi
        w0, w1 = w1, wi

    return res

def chebyshev_solver_main(A_op, b, lanczos_steps=30, chebyshev_degree=20):
    lambda_min, lambda_max = estimate_eigenvalues(A_op, n_steps=lanczos_steps)

    # --- FIXED PADDING LOGIC ---
    # Pad the upper bound comfortably (linear functions are safe here)
    lambda_max = lambda_max * 1.05

    # Pad the lower bound RELATIVE to itself.
    # Do NOT subtract a fraction of the total width.
    # We want a margin, but we must stay away from the singularity at 0.
    lambda_min = lambda_min * 0.9
    # ---------------------------

    # Add a small shift to avoid singularity (and ensure min > 0)
    lambda_min = max(lambda_min, 1e-6)

    x = chebyshev_sqrt_solver(A_op, b, lambda_min, lambda_max, degree=chebyshev_degree)
    return x

if __name__ == '__main__':
    # Create a synthetic SPD matrix
    n = 100
    np.random.seed(0)
    A = np.random.rand(n, n)
    A = A @ A.T + np.eye(n) * 0.1 # Ensure SPD

    # Create a random vector b
    b = np.random.rand(n)

    # Define the matrix-vector product function
    def A_mv(v):
        return A @ v

    A_op = LinearOperator((n, n), matvec=A_mv)

    # --- Parameters ---
    lanczos_steps = 30
    chebyshev_degree = 30

    # --- Run Solvers ---
    x_cheby = chebyshev_solver_main(A_op, b, lanczos_steps, chebyshev_degree)
    x_cg = lanczos_cg_solver(A_op, b, lanczos_steps)

    # --- Baseline (True Solution) ---
    from scipy.linalg import sqrtm
    A_sqrt_inv = np.linalg.inv(sqrtm(A))
    x_true = A_sqrt_inv @ b

    # --- Error Calculation ---
    error_cheby = np.linalg.norm(x_cheby - x_true) / np.linalg.norm(x_true)
    error_cg = np.linalg.norm(x_cg - x_true) / np.linalg.norm(x_true)
    print(f"Relative error of Chebyshev solver: {error_cheby:.6f}")
    print(f"Relative error of Lanczos-CG solver: {error_cg:.6f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(x_true, label='True Solution', linestyle='--')
    plt.plot(x_cheby, label=f'Chebyshev (err={error_cheby:.4f})', alpha=0.8)
    plt.plot(x_cg, label=f'Lanczos-CG (err={error_cg:.4f})', alpha=0.6)
    plt.title('Comparison of Solver Solutions')
    plt.xlabel('Vector Component Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'solution_comparison.png'))
    plt.show()
