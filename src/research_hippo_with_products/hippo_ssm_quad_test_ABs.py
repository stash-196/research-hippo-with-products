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
N = 4
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
A_infty_exp = expm(A_gen)
print("A_gen :\n", np.round(A_gen, 5))
print(
    "-(-A_gen^T+I) :\n",
    -(-np.round(A_gen.T, 3) + np.eye(N)),
    5,
)
print("A_infty_exp :\n", np.round(A_infty_exp, 5))


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
dt = 1.0
I = np.eye(N)
inv = np.linalg.inv(I - dt / 2 * A_cont)
dA = inv @ (I + dt / 2 * A_cont)
dB = inv @ (dt * B_cont)
print("Discretized A:\n", np.round(dA, 3))
print("Discretized B:\n", np.round(dB.flatten(), 3))


# %%
# Compute A_infty
A_infty = np.zeros((N, N))
for n in tqdm(range(N), desc="Computing A_infty via Direct Integration"):
    for m in range(N):

        def integrand(z):
            # The lag operator is applied to the second basis function
            return phi(n, z) * phi(m, np.exp(-1) * z)

        A_infty[n, m] = quad(integrand, 0, 1)[0]

print("A_infty:\n", np.round(A_infty, 5))


# %%
import numpy as np
from scipy.special import roots_legendre, eval_legendre
import time


# --- Vectorized Setup ---
def get_gauss_quadrature(order, a=0, b=1):
    """Get Gauss-Legendre nodes and weights on interval [a, b]."""
    x, w = roots_legendre(order)  # On [-1, 1]
    z = a + (b - a) * (x + 1) / 2
    w = w * (b - a) / 2
    return z, w


def vectorized_phi(phi_func, n_values, z_values):
    """
    Evaluates a given basis function phi_n(z) for all n in n_values and z in z_values in a vectorized way.

    Args:
    phi_func (callable): Function phi(n, z) returning the basis value for index n at point z.
    n_values (1D array): Array of basis indices [0, 1, ..., N-1].
    z_values (1D array): Array of points to evaluate at [z_0, ..., z_k-1].

    Returns:
    (k, N) array where result[i, j] = phi_func(j, z_values[i]).
    """
    k = len(z_values)
    N = len(n_values)
    phi_vals = np.zeros((k, N))
    for j, n in enumerate(n_values):
        phi_vals[:, j] = phi_func(n, z_values)
    return phi_vals


# %%
# Lag operator transition matrices
def lag_transition_infty(N, dt=1.0, quad_order=None):
    if quad_order is None:
        quad_order = (
            2 * N
        )  # Exact for degree < 2N (safe for Legendre products up to degree 2N-2)

    # Direct computation of A_infty via quadrature on [0,1]
    z_nodes_a, z_weights_a = get_gauss_quadrature(quad_order, 0, 1)
    n_vals = np.arange(N)
    eval_n = vectorized_phi(phi, n_vals, z_nodes_a)  # (quad_order, N)
    warped_z = np.exp(-dt) * z_nodes_a
    eval_m = vectorized_phi(phi, n_vals, warped_z)  # (quad_order, N)

    # Fix: Sum over quadrature points to get (N, N)
    A_infty_vec = np.einsum("i,ik,il->kl", z_weights_a, eval_n, eval_m)
    start_time = time.time()
    print(f"\nA_infty computed via vectorized Gauss quadrature (order {quad_order}):")
    print(np.round(A_infty_vec, 3))
    end_time = time.time()
    print(f"Total computation time for A_infty: {end_time - start_time:.2f} seconds")

    # Compute B via quadrature on [exp(-dt), 1]
    z_min = np.exp(-dt)
    z_max = 1.0
    z_nodes_b, z_weights_b = get_gauss_quadrature(quad_order, z_min, z_max)
    basis_evals_b = vectorized_phi(phi, n_vals, z_nodes_b)  # (quad_order, N)

    # B = basis_evals_b.T @ z_weights_b (N, quad_order) @ (quad_order,) -> (N,)
    B_zoh_vec = basis_evals_b.T @ z_weights_b

    start_time = time.time()
    print(f"\nB computed via vectorized Gauss quadrature:")
    print(np.round(B_zoh_vec, 3))
    end_time = time.time()
    print(f"Total computation time for B: {end_time - start_time:.2f} seconds")

    return A_infty_vec, B_zoh_vec


N = 333
A_infty, B_zoh = lag_transition_infty(N)
print("A_infty:\n", np.round(A_infty, 5))
print("B_zoh:\n", np.round(B_zoh, 5))

c_one_hot = np.zeros(N)
c_one_hot[0] = 1.0
print("c_one_hot:\n", c_one_hot)

c_new = A_infty @ c_one_hot
print("A_infty @ c_one_hot:\n", c_new)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
from tqdm import tqdm
from scipy.special import legendre, roots_legendre, eval_legendre
import time
import matplotlib.pyplot as plt


# Basis function (Normalized Legendre on [0,1])
def phi(n, z):
    y = 2 * z - 1
    return np.sqrt(2 * n + 1) * legendre(n)(y)


# Derivative of phi (for Legendre)
def phi_prime(n, z):
    y = 2 * z - 1
    p_n = legendre(n)
    p_prime_n = p_n.deriv()
    return np.sqrt(2 * n + 1) * p_prime_n(y) * 2  # Chain rule dy/dz = 2


# Warp functions for HiPPO-LegS
def sigma(t, s):
    return np.exp(s - t)


def sigma_inv(t, z):
    return t + np.log(z)


def get_gauss_quadrature(order, a=0, b=1):
    """Get Gauss-Legendre nodes and weights on interval [a, b]."""
    x, w = roots_legendre(order)
    z = a + (b - a) * (x + 1) / 2
    w = w * (b - a) / 2
    return z, w


# Generic vectorized evaluator
def vectorized_basis(basis_func, n_values, z_values):
    k = len(z_values)
    N = len(n_values)
    vals = np.zeros((k, N))
    for j, n in enumerate(n_values):
        vals[:, j] = basis_func(n, z_values)
    return vals


# Lag operator transition matrices
def lag_transition(N, dt=1.0, quad_order=None):
    if quad_order is None:
        quad_order = 2 * N

    n_vals = np.arange(N)

    # Compute A_gen via quadrature on [0,1]
    z_nodes, z_weights = get_gauss_quadrature(quad_order, 0, 1)
    phi_vals = vectorized_basis(phi, n_vals, z_nodes)  # (k, N)
    phi_prime_vals = vectorized_basis(phi_prime, n_vals, z_nodes)  # (k, N)
    A_gen = np.einsum(
        "i,ik,il->kl", z_weights, -phi_vals, z_nodes[:, None] * phi_prime_vals
    )

    start_time = time.time()
    print(f"\nA_gen computed via vectorized Gauss quadrature (order {quad_order}):")
    print(np.round(A_gen, 3))
    end_time = time.time()
    print(f"Total computation time for A_gen: {end_time - start_time:.2f} seconds")

    # A_infty = expm(dt * A_gen)  # If needed

    # Compute B via quadrature on [exp(-dt), 1]
    z_min = np.exp(-dt)
    z_max = 1.0
    z_nodes_b, z_weights_b = get_gauss_quadrature(quad_order, z_min, z_max)
    basis_evals_b = vectorized_basis(phi, n_vals, z_nodes_b)  # (k, N)
    B_zoh_vec = basis_evals_b.T @ z_weights_b

    start_time = time.time()
    print(f"\nB computed via vectorized Gauss quadrature:")
    print(np.round(B_zoh_vec, 3))
    end_time = time.time()
    print(f"Total computation time for B: {end_time - start_time:.2f} seconds")

    return A_gen, B_zoh_vec  # Return A_gen for discretization


A_gen, B_zoh = lag_transition(N=4, dt=1.0)
print("A_gen:\n", np.round(A_gen, 5))
print("B_zoh:\n", np.round(B_zoh, 5))


# %%
class LagHiPPO(nn.Module):
    """Lag Operator-based time invariant x_{t+1} = A x_t + B u_t"""

    def __init__(self, N, dt=1.0, T=1.0, c=0.0):
        super().__init__()
        self.N = N
        self.dt = dt
        self.T = T
        self.c = c

        time_start = time.time()
        dA, dB = lag_transition(N, dt=self.dt)
        time_end = time.time()
        print(
            f"LagHiPPO Total computation time for lag_transition: {time_end - time_start:.2f} seconds"
        )

        self.register_buffer("dA", torch.tensor(dA, dtype=torch.float64))  # (N, N)
        self.register_buffer("dB", torch.tensor(dB, dtype=torch.float64))  # (N,)

        time_start = time.time()
        self.vals = np.arange(0.0, T, dt)
        self.eval_matrix = self._compute_eval_matrix()
        time_end = time.time()
        print(
            f"LagHiPPO Total computation time for eval_matrix: {time_end - time_start:.2f} seconds"
        )

        time_start = time.time()
        self.measure = self._compute_measure()
        time_end = time.time()
        print(
            f"LagHiPPO Total computation time for measure: {time_end - time_start:.2f} seconds"
        )

    def _compute_eval_matrix(self):
        z_warped = np.exp(-self.vals)
        n_vals = np.arange(self.N)
        eval_matrix = vectorized_phi(phi, n_vals, z_warped)
        decay_factor = np.exp(-self.c * self.vals)
        eval_matrix = eval_matrix * decay_factor[:, np.newaxis]
        return torch.tensor(eval_matrix, dtype=torch.float64)

    def _compute_measure(self):
        return np.exp(-self.vals)

    def forward(self, inputs, fast=True):
        inputs = inputs.unsqueeze(-1)
        u = inputs * self.dB

        if fast:
            dA = self.dA.unsqueeze(0).repeat(u.size(0), 1, 1)
            return variable_unroll_matrix(dA, u)

        c = torch.zeros(u.shape[1:], device=inputs.device, dtype=torch.float64)
        cs = []
        for f in inputs:
            c = torch.matmul(self.dA, c.unsqueeze(-1)).squeeze(-1) + self.dB * f
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c, evals=None):
        if evals is not None:
            z_warped = np.exp(-evals)
            n_vals = np.arange(self.N)
            eval_matrix = vectorized_phi(phi, n_vals, z_warped)
            decay_factor = np.exp(-self.c * evals)
            eval_matrix = eval_matrix * decay_factor[:, np.newaxis]
            eval_matrix = torch.tensor(eval_matrix, dtype=torch.float64)
        else:
            eval_matrix = self.eval_matrix

        if self.measure is not None:
            m = self.measure[self.measure != 0.0]
            eval_matrix = eval_matrix[: len(m), :]

        c = c.unsqueeze(-1).to(dtype=torch.float64)  # Ensure c is float64
        y = torch.matmul(eval_matrix.to(c.device), c)
        return y.squeeze(-1).flip(-1)


# Test code
if __name__ == "__main__":
    N = 4
    dt = 1.0
    T = 10.0
    model = LagHiPPO(N=N, dt=dt, T=T)
    A_infty = model.dA.numpy()

# %%
N = 333
dt = 1e-3
T = 10.0
model = LagHiPPO(N=N, dt=dt, T=T)
A_infty = model.dA.numpy()


for n in range(N)[-5:-1]:
    c_one_hot = np.zeros(N, dtype=np.float64)
    c_one_hot[n] = 1.0
    recon_original = model.reconstruct(
        torch.tensor(c_one_hot, dtype=torch.float64)
    ).numpy()
    c_new = A_infty @ c_one_hot
    recon_shifted = model.reconstruct(torch.tensor(c_new, dtype=torch.float64)).numpy()

    plt.figure()
    plt.plot(model.vals, recon_original, label="Original")
    plt.plot(model.vals, recon_shifted, label="Shifted")
    plt.title(f"Basis n={n}")
    plt.legend()
    plt.show()
    plt.close()

    zoom_last_vals = 20
    zoom_last_vals = T / dt if T / dt < zoom_last_vals else zoom_last_vals
    plt.figure()
    plt.plot(
        model.vals[-zoom_last_vals:], recon_original[-zoom_last_vals:], label="Original"
    )
    plt.plot(
        model.vals[-zoom_last_vals:], recon_shifted[-zoom_last_vals:], label="Shifted"
    )
    plt.title(f"Basis n={n} zoomed in")
    plt.legend()
    plt.show()
    plt.close()
# %%
