# %%
"""""
Investigate and test numerical integration for HiPPO-LegS.
Compare with analytical solution.
Results: A_HiPPO = -(-A_gen^T + I)
B_HiPPO =/= B_ZOH, dB =/= B_ZOH
""" ""

# %%
# import numpy as np
from scipy.linalg import expm
from scipy.special import legendre
from scipy.integrate import quad
from einops import repeat
import numpy as np
from tqdm import tqdm
import time


# Basis: Normalized Legendre on [0,1]
def phi(n, z):
    y = 2 * z - 1  # Map z from [0,1] to [-1,1]
    return np.sqrt(2 * n + 1) * legendre(n)(y)


# Derivative of phi (for Legendre)
def phi_prime(n, z):
    y = 2 * z - 1
    p_n = legendre(n)
    p_prime_n = p_n.deriv()
    return np.sqrt(2 * n + 1) * p_prime_n(y) * 2  # Chain rule dy/dz = 2


# Warp: Exponential for HiPPO-LegS on [0,1]
def sigma(t, s):
    return np.exp(s - t)


# Inverse warp
def sigma_inv(t, z):
    return t + np.log(z)


# Lag operator
def lag(t, z):
    return sigma(t + 1, sigma_inv(t, z))


# Test orthonormality
def test_orthonormality(N=4, phi=phi):
    for n in range(N):
        for m in range(N):

            def integrand(z):
                return phi(n, z) * phi(m, z)

            result = quad(integrand, 0, 1)[0]
            if n == m:
                assert np.isclose(result, 1), f"Failed at n={n}, m={m}, got {result}"
            else:
                assert np.isclose(result, 0), f"Failed at n={n}, m={m}, got {result}"
    print("Orthonormality test passed!")


test_orthonormality()

# %%
N = 4
t = 10  # Arbitrary, constant for this warp
Z_min, Z_max = 0, 1

# Compute A_t
A_t = np.zeros((N, N))
start_time = time.time()

# Using tqdm to create progress bars for both loops
for n in tqdm(range(N), desc="Outer loop"):
    for m in tqdm(range(N), desc=f"Inner loop (n={n})", leave=False):

        def integrand(z):
            return phi(n, lag(t, z)) * phi(m, z)

        A_t[n, m] = quad(integrand, Z_min, Z_max)[0]

end_time = time.time()
print(f"\nTotal computation time: {end_time - start_time:.2f} seconds")
print("Computed A_t:\n", np.round(A_t, 3))

# Compute B_t
B_t = np.zeros(N)
start_time = time.time()
for n in tqdm(range(N), desc="Computing B"):

    def integrand(z):
        return phi(n, z)

    z_min = sigma(t + 1, t)
    z_max = sigma(t + 1, t + 1)
    B_t[n] = quad(integrand, z_min, z_max)[0]

end_time = time.time()
print(f"\nTotal computation time for B: {end_time - start_time:.2f} seconds")
print("Computed B_t:\n", np.round(B_t, 3))


# %%
# Compute Analytical A and B for HiPPO-LegS
def transition(measure, N):
    if measure == "legs":
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy()
    return A, B


N = 4
A_cont, B_cont = transition("legs", N)
print("Continuous A:\n", np.round(A_cont, 3))
print("Continuous B:\n", np.round(B_cont.flatten(), 3))
dt = 1e-3
I = np.eye(N)
inv = np.linalg.inv(I - dt / 2 * A_cont)
dA = inv @ (I + dt / 2 * A_cont)
dB = inv @ (dt * B_cont)
print("Discretized A:\n", np.round(dA, 3))
print("Discretized B:\n", np.round(dB.flatten(), 3))


# %%
# Compute A_deriv without signs
A_gen = np.zeros((N, N))
start_time = time.time()
for n in tqdm(range(N), desc="Outer loop for A_gen"):
    for m in tqdm(range(N), desc=f"Inner loop (n={n})", leave=False):

        def integrand(z):
            p_m = legendre(m)
            p_prime_m = p_m.deriv()
            y = 2 * z - 1
            alpha = np.sqrt(2 * m + 1)
            phi_m_prime = alpha * p_prime_m(y) * 2
            phi_n_val = phi(n, z)
            return -phi_n_val * z * phi_m_prime

        A_gen[n, m] = quad(integrand, Z_min, Z_max, limit=100)[0]
end_time = time.time()
print("\nTotal computation time for A_gen:", end_time - start_time)
print("A_gen :\n", np.round(A_gen, 3))
print(
    "-(-A_gen^T+I) :\n",
    -(-np.round(A_gen.T, 3) + np.eye(N)),
    3,
)
A_infty_exp = expm(A_gen * dt)
print("A_infty_exp :\n", np.round(A_infty_exp, 5))


# %%
# Compute A_infty
A_infty = np.zeros((N, N))
for n in tqdm(range(N), desc="Computing A_infty via Direct Integration"):
    for m in range(N):

        def integrand(z):
            # The lag operator is applied to the second basis function
            return phi(n, z) * phi(m, np.exp(-dt) * z)

        A_infty[n, m] = quad(integrand, 0, 1)[0]

print("A_infty:\n", np.round(A_infty, 5))


# %%
import numpy as np
from scipy.special import roots_legendre, eval_legendre
import time

# --- Vectorized Setup ---


def get_gauss_quadrature(k, a, b):
    """
    Get Gaussian quadrature nodes and weights for the interval [a, b].
    """
    # Get nodes and weights for the canonical interval [-1, 1]
    nodes, weights = roots_legendre(k)
    # Rescale to the interval [a, b]
    nodes_rescaled = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    weights_rescaled = 0.5 * (b - a) * weights
    return nodes_rescaled, weights_rescaled


def vectorized_phi(n_values, z_values):
    """
    Evaluates all basis functions on all z values in a vectorized way.

    Args:
        n_values (1D array): Array of basis indices [0, 1, ..., N-1].
        z_values (1D array): Array of points to evaluate at [z_0, ..., z_k-1].

    Returns:
        (k, N) array where result[i, j] = phi(j, z_values[i]).
    """
    y = 2 * z_values - 1  # Map z from [0, 1] to [-1, 1]
    # eval_legendre is vectorized over both arguments
    legendre_evals = eval_legendre(n_values[np.newaxis, :], y[:, np.newaxis])
    normalization = np.sqrt(2 * n_values + 1)
    return normalization * legendre_evals


# %%
# --- Parameters ---
N = 4  # Let's use a more realistic size
k = 100  # Number of quadrature points (higher k = more accuracy)

n_vals = np.arange(N)

# --- 1. Compute B_ZOH (Vectorized) ---
start_time = time.time()

# Get nodes and weights for the interval [e^-1, 1]
z_nodes_b, z_weights_b = get_gauss_quadrature(k, np.exp(-1), 1)

# Evaluate all phi_n at all nodes for B
# Shape: (k, N)
basis_evals_b = vectorized_phi(n_vals, z_nodes_b)

# The integral is now a simple matrix-vector product (dot product)
# (N, k) @ (k,) -> (N,)
B_zoh_vec = basis_evals_b.T @ z_weights_b

end_time = time.time()
print(f"Vectorized B_ZOH computation time: {end_time - start_time:.6f} seconds")


# --- 2. Compute A_infty (Vectorized) ---
start_time = time.time()

# Get nodes and weights for the interval [0, 1]
z_nodes_a, z_weights_a = get_gauss_quadrature(k, 0, 1)

# Evaluate phi_n(z) at the nodes. Shape: (k, N)
eval_n = vectorized_phi(n_vals, z_nodes_a)

# Evaluate phi_m(e^-1 * z) at the warped nodes. Shape: (k, N)
eval_m = vectorized_phi(n_vals, np.exp(-1) * z_nodes_a)

# The integral is now a series of matrix multiplications
# A[n, m] = sum(w[i] * eval_n[i, n] * eval_m[i, m]) over i
# This is equivalent to: eval_n.T @ diag(w) @ eval_m
A_infty_vec = (eval_n.T * z_weights_a) @ eval_m

end_time = time.time()
print(f"Vectorized A_infty computation time: {end_time - start_time:.6f} seconds")

# --- Display Results (optional) ---
print("\nComputed B_ZOH (first 5 elements):\n", np.round(B_zoh_vec[:5], 4))
print("\nComputed A_infty (top-left 5x5):\n", np.round(A_infty_vec[:5, :5], 5))
# %%

# %%
# Lag operator transition matrices
# def lag_transition(N, dt=1.0, quad_order=None):
#     if quad_order is None:
#         quad_order = 20 * N

#     n_vals = np.arange(N)

#     # Compute A_gen via quadrature on [0,1]
#     z_nodes, z_weights = get_gauss_quadrature(quad_order, 0, 1)
#     phi_vals = vectorized_basis(phi, n_vals, z_nodes)  # (k, N)
#     phi_prime_vals = vectorized_basis(phi_deriv, n_vals, z_nodes)  # (k, N)
#     A_gen = np.einsum(
#         "i,ik,il->kl", z_weights, -phi_vals, z_nodes[:, None] * phi_prime_vals
#     )

#     start_time = time.time()
#     print(f"\nA_gen computed via vectorized Gauss quadrature (order {quad_order}):")
#     print(np.round(A_gen, 3))
#     end_time = time.time()
#     print(f"Total computation time for A_gen: {end_time - start_time:.2f} seconds")

#     # A_infty = expm(dt * A_gen)  # If needed

#     # Compute B via quadrature on [exp(-dt), 1]
#     z_min = np.exp(-dt)
#     z_max = 1.0
#     z_nodes_b, z_weights_b = get_gauss_quadrature(quad_order, z_min, z_max)
#     basis_evals_b = vectorized_basis(phi, n_vals, z_nodes_b)  # (k, N)
#     B_zoh_vec = basis_evals_b.T @ z_weights_b

#     start_time = time.time()
#     print(f"\nB computed via vectorized Gauss quadrature:")
#     print(np.round(B_zoh_vec, 3))
#     end_time = time.time()
#     print(f"Total computation time for B: {end_time - start_time:.2f} seconds")

#     return A_gen, B_zoh_vec  # Return A_gen for discretization
