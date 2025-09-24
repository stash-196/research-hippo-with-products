# %% Lag HiPPO
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import expm
from scipy import signal
from scipy.integrate import quad
from tqdm import tqdm
from scipy.special import legendre, eval_legendre, roots_legendre
import time

from research_hippo_with_products.visualizers import (
    plot_basis_functions,
    plot_reconstructed_signal,
    plot_signal,
)


def shift_up(a, s=None, drop=True, dim=0):
    assert dim == 0
    if s is None:
        s = torch.zeros_like(a[0, ...])
    s = s.unsqueeze(dim)
    if drop:
        a = a[:-1, ...]
    return torch.cat((s, a), dim=dim)


def batch_mult(A, u, has_batch=None):
    """Matrix mult A @ u with special case to save memory if u has additional batch dim

    The batch dimension is assumed to be the second dimension
    A : (L, ..., N, N)
    u : (L, [B], ..., N)
    has_batch: True, False, or None. If None, determined automatically

    Output:
    x : (L, [B], ..., N)
      A @ u broadcasted appropriately
    """

    if has_batch is None:
        has_batch = len(u.shape) >= len(A.shape)

    if has_batch:
        u = u.permute([0] + list(range(2, len(u.shape))) + [1])
    else:
        u = u.unsqueeze(-1)
    v = A @ u
    if has_batch:
        v = v.permute([0] + [len(u.shape) - 1] + list(range(1, len(u.shape) - 1)))
    else:
        v = v[..., 0]
    return v


def interleave(a, b, uneven=False, dim=0):
    """Interleave two tensors of same shape"""
    # assert(a.shape == b.shape)
    assert dim == 0  # TODO temporary to make handling uneven case easier
    if dim < 0:
        dim = N + dim
    if uneven:
        a_ = a[-1:, ...]
        a = a[:-1, ...]
    c = torch.stack((a, b), dim + 1)
    out_shape = list(a.shape)
    out_shape[dim] *= 2
    c = c.view(out_shape)
    if uneven:
        c = torch.cat((c, a_), dim=dim)
    return c


def variable_unroll_general_sequential(A, u, s, op, variable=True):
    """Unroll with variable (in time/length) transitions A with general associative operation

    A : ([L], ..., N, N) dimension L should exist iff variable is True
    u : (L, [B], ..., N) updates
    s : ([B], ..., N) start state
    output : x (..., N)
    x[i, ...] = A[i]..A[0] s + A[i..1] u[0] + ... + A[i] u[i-1] + u[i]
    """

    if not variable:
        A = A.expand((u.shape[0],) + A.shape)

    outputs = []
    for A_, u_ in zip(torch.unbind(A, dim=0), torch.unbind(u, dim=0)):
        s = op(A_, s)
        s = s + u_
        outputs.append(s)

    output = torch.stack(outputs, dim=0)
    return output


def variable_unroll_general(
    A, u, s, op, compose_op=None, sequential_op=None, variable=True, recurse_limit=16
):
    """Bottom-up divide-and-conquer version of variable_unroll.

    compose is an optional function that defines how to compose A without multiplying by a leaf u
    """

    if u.shape[0] <= recurse_limit:
        if sequential_op is None:
            sequential_op = op
        return variable_unroll_general_sequential(A, u, s, sequential_op, variable)

    if compose_op is None:
        compose_op = op

    uneven = u.shape[0] % 2 == 1
    has_batch = len(u.shape) >= len(A.shape)

    u_0 = u[0::2, ...]
    u_1 = u[1::2, ...]

    if variable:
        A_0 = A[0::2, ...]
        A_1 = A[1::2, ...]
    else:
        A_0 = A
        A_1 = A

    u_0_ = u_0
    A_0_ = A_0
    if uneven:
        u_0_ = u_0[:-1, ...]
        if variable:
            A_0_ = A_0[:-1, ...]

    u_10 = op(A_1, u_0_)  # batch_mult(A_1, u_0_, has_batch)
    u_10 = u_10 + u_1
    A_10 = compose_op(A_1, A_0_)

    # Recursive call
    x_1 = variable_unroll_general(
        A_10,
        u_10,
        s,
        op,
        compose_op,
        sequential_op,
        variable=variable,
        recurse_limit=recurse_limit,
    )

    x_0 = shift_up(x_1, s, drop=not uneven)
    x_0 = op(A_0, x_0)  # batch_mult(A_0, x_0, has_batch)
    x_0 = x_0 + u_0

    x = interleave(
        x_0, x_1, uneven, dim=0
    )  # For some reason this interleave is slower than in the (non-variable) unroll_recursive
    return x


def variable_unroll_matrix(A, u, s=None, variable=True, recurse_limit=16):
    if s is None:
        s = torch.zeros_like(u[0])
    has_batch = len(u.shape) >= len(A.shape)
    op = lambda x, y: batch_mult(x, y, has_batch)
    sequential_op = lambda x, y: batch_mult(x.unsqueeze(0), y.unsqueeze(0), has_batch)[
        0
    ]
    matmul = lambda x, y: x @ y
    return variable_unroll_general(
        A,
        u,
        s,
        op,
        compose_op=matmul,
        sequential_op=sequential_op,
        variable=variable,
        recurse_limit=recurse_limit,
    )


# %%
# Real Fourier basis on [0,1]
def phi_fourier(n, z):
    """
    Computes the n-th real Fourier basis function on [0,1].
    Uses 2*pi for correct periodicity on the unit interval.
    """
    if n == 0:
        return np.ones_like(z)
    # k starts from 1
    k = (n // 2) + 1 if n % 2 == 0 else (n + 1) // 2

    if n % 2 == 1:  # Odd n > 0 for cosine terms
        return np.sqrt(2) * np.cos(2 * np.pi * k * z)
    else:  # Even n > 0 for sine terms
        return np.sqrt(2) * np.sin(2 * np.pi * k * z)


def phi_fourier_deriv(n, z):
    """
    Computes the derivative of the n-th real Fourier basis function on [0,1].
    """
    if n == 0:
        return np.zeros_like(z)
    # k starts from 1
    k = (n // 2) + 1 if n % 2 == 0 else (n + 1) // 2

    if n % 2 == 1:  # Derivative of cosine is -sine
        return -2 * np.pi * k * np.sqrt(2) * np.sin(2 * np.pi * k * z)
    else:  # Derivative of sine is cosine
        return 2 * np.pi * k * np.sqrt(2) * np.cos(2 * np.pi * k * z)


# Warp functions for HiPPO-LegS
def sigma(t, s):
    return np.exp(s - t)


def sigma_deriv(t, s):
    return np.exp(s - t)


def sigma_uniform(t, s, T=1.0):
    return (s - (t - T)) / T  # Shift to positive, normalize if needed


def sigma_uniform_deriv(t, s, T=1.0):
    return 1.0 / T


# def sigma_uniform(t, s, dt):
#     return (s - t + 1.0)  # Shift to positive, normalize if needed

# def sigma_uniform_deriv(t, s):
#     return 1.0


def phi(n, z):
    y = 2 * z - 1
    return np.sqrt(2 * n + 1) * eval_legendre(n, y)


def phi_deriv(n, z):
    y = 2 * z - 1
    if n == 0:
        return np.zeros_like(z)
    p_n = eval_legendre(n, y)
    p_nm1 = eval_legendre(n - 1, y)
    den = 1.0 - y**2
    result = np.zeros_like(z)
    nonzero_den = den > 1e-10
    result[nonzero_den] = (
        n
        / den[nonzero_den]
        * (p_nm1[nonzero_den] - y[nonzero_den] * p_n[nonzero_den])
        * 2.0
    )
    zero_den = den <= 1e-10
    signs = np.where(y[zero_den] > 0, 1.0, (-1.0) ** (n + 1))
    result[zero_den] = signs * n * (n + 1) / 2 * 2.0
    return np.sqrt(2 * n + 1) * result


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


def lag_transition(N, phi_func, phi_deriv_func, dt=1.0, quad_order=None):
    if quad_order is None:
        quad_order = max(2 * N, 4 * N if N > 20 else 2 * N)  #

    n_vals = np.arange(N)

    # Compute A_gen via quadrature on [0,1]
    z_nodes, z_weights = get_gauss_quadrature(quad_order, 0, 1)
    phi_vals = vectorized_basis(phi_func, n_vals, z_nodes)  # (k, N)
    phi_deriv_vals = vectorized_basis(phi_deriv_func, n_vals, z_nodes)  # (k, N)
    A_gen = np.einsum(
        # "i,ik,il->kl", z_weights, -phi_vals, z_nodes[:, None] * phi_deriv_vals
        "i,ik,il->kl",
        z_weights,
        phi_vals,
        z_nodes[:, None] * phi_deriv_vals,
    )

    start_time = time.time()
    print(f"\nA_gen computed via vectorized Gauss quadrature (order {quad_order}):")
    print(np.round(A_gen, 3))
    end_time = time.time()
    print(f"Total computation time for A_gen: {end_time - start_time:.2f} seconds")

    # A_infty = expm(dt * A_gen)  # If needed

    # Compute B via phi(1)
    B_delta = np.zeros(N)
    for n in range(N):
        B_delta[n] = phi_func(n, 1)
    return A_gen, B_delta  # Return A_gen for discretization


class LagHiPPO(nn.Module):
    """Lag Operator-based time invariant x_{t+1} = A x_t + B u_t"""

    def __init__(
        self,
        N,
        phi_func,
        phi_deriv_func,
        sigma_func,
        sigma_deriv_func,
        dt=1.0,
        T=1.0,
        c=0.0,
        discretization="bilinear",
        mode="scale",
        window=1.0,
    ):
        """
                N: the order of the HiPPO projection
                dt: discretization step size
                T: total time for evaluation matrix
                c: optional decay parameter
        mode: 'scale' for exponential/compressive warps, 'translate' for uniform/truncated warps.
                window: For 'translate' mode, the fixed truncation window (normalized to 1 by default to match HiPPO-LegT).
        """
        super().__init__()
        self.N = N
        self.dt = dt
        self.T = T
        self.c = c
        self.phi_func = phi_func
        self.phi_deriv_func = phi_deriv_func
        self.sigma_func = sigma_func
        self.sigma_deriv_func = sigma_deriv_func
        self.mode = mode
        self.window = window

        # Compute lag-based transition matrices
        A_gen, B = lag_transition(N, phi_func, phi_deriv_func, dt=self.dt)

        A_HiPPO = -(A_gen + np.eye(N)).T

        self.A = A_HiPPO
        self.B = B
        self.measure_fn = self.measure()

        C = np.ones((1, N))
        D = np.zeros((1,))
        dA, dB, _, _, _ = signal.cont2discrete(
            (A_HiPPO, B, C, D), dt=dt, method=discretization
        )

        dB = dB

        self.register_buffer("dA", torch.Tensor(dA))  # (N, N)
        self.register_buffer("dB", torch.Tensor(dB))  # (N,)

        # Evaluation matrix for reconstruction
        self.vals = np.arange(0.0, T, dt)
        self.eval_matrix = self._compute_eval_matrix()  # (len(vals), N)
        self.measure = self._compute_measure()  # Using original HiPPO convention

    def _compute_eval_matrix(self):
        start_time = time.time()
        z_warped = self.sigma_func(0, -self.vals)  # (len(vals),)
        n_vals = np.arange(self.N)
        eval_matrix = vectorized_basis(
            self.phi_func, n_vals, z_warped
        )  # (len(vals), N)
        decay_factor = np.exp(-self.c * self.vals)  # (len(vals),)
        eval_matrix = eval_matrix * decay_factor[:, np.newaxis]  # Broadcast decay to
        end_time = time.time()
        print(
            f"LagHiPPO: Total computation time for basis: {end_time - start_time:.2f} seconds"
        )

        return torch.Tensor(eval_matrix)

    def measure(self):
        return lambda x: self.sigma_deriv_func(0, -x)

    def _compute_measure(self):
        return self.measure()(self.vals)

    def forward(self, inputs, fast=True):
        """
        inputs : (length, ...)
        output : (length, ..., N)
        """
        time_start = time.time()
        inputs = inputs.unsqueeze(-1)
        u = inputs * self.dB  # (length, ..., N)

        if fast:
            dA = self.dA.unsqueeze(0).repeat(u.size(0), 1, 1)  # (L, N, N)
            result = variable_unroll_matrix(dA, u)
            time_end = time.time()
            print(
                f"LagHiPPO: Total computation time for fast forward pass: {time_end - time_start:.2f} seconds"
            )
            return result

        c = torch.zeros(u.shape[1:], device=inputs.device)
        cs = []
        for f in inputs:
            c = torch.matmul(self.dA, c.unsqueeze(-1)).squeeze(-1) + self.dB * f
            cs.append(c)
        time_end = time.time()
        print(
            f"LagHiPPO: Total computation time for forward pass: {time_end - time_start:.2f} seconds"
        )
        return torch.stack(cs, dim=0)

    def reconstruct(self, c, evals=None):
        """
        c: (..., N,) HiPPO coefficients
        output: (..., L,)
        """
        if evals is not None:
            z_warped = self.sigma_func(0, -evals)
            eval_matrix = np.zeros((len(evals), self.N))
            time_start = time.time()
            eval_matrix = self._compute_eval_matrix()
            time_end = time.time()
            print(
                f"LagHiPPO: reconstruct() Total computation time for eval_matrix: {time_end - time_start:.2f} seconds"
            )
        else:
            eval_matrix = self.eval_matrix

        # if self.measure is not None:
        #     m = self.measure[self.measure != 0.0]
        #     eval_matrix = eval_matrix[:len(m), :]

        c = c.unsqueeze(-1)
        y = torch.matmul(eval_matrix.to(c.device), c)
        return y.squeeze(-1).flip(-1)


def periodic_signal(vals):
    """A sum of sine waves, perfect for testing the Fourier basis."""
    return np.sin(2 * np.pi * 2 * vals) + 0.5 * np.sin(2 * np.pi * 5 * vals)


# %%
N = 64
T = 10
dt = 0.01
vals = np.arange(0.0, T, dt)
u = torch.tensor(
    periodic_signal(vals / T), dtype=torch.float
)  # Normalize time for signal

phi_func = phi
phi_deriv_func = phi_deriv
sigma_func = sigma
sigma_deriv_func = sigma_deriv
hippo_lag = LagHiPPO(
    N=N,
    phi_func=phi_func,
    phi_deriv_func=phi_deriv_func,
    sigma_func=sigma_func,
    sigma_deriv_func=sigma_deriv_func,
    dt=dt,
    T=T,
    mode="translate",
)
u_hippo = hippo_lag.reconstruct(hippo_lag(u)[-1]).cpu()


# %%
PAPER_READY_PLOTS = True
plot_basis_functions(
    x=vals,
    Bases=hippo_lag.eval_matrix.numpy(),
    ks=range(N),
    data_name="LagHiPPO",
    domain_name="s",
    basis_type="legendre",
    num_functions_front=None,
    num_functions_back=None,
    title="LagHiPPO Basis Functions",
    paper_ready=PAPER_READY_PLOTS,
)
plot_basis_functions(
    x=sigma_func(0, -vals),
    Bases=hippo_lag.eval_matrix.numpy(),
    ks=range(N),
    data_name="LagHiPPO",
    domain_name="z",
    basis_type="legendre",
    num_functions_front=None,
    num_functions_back=None,
    title="LagHiPPO Basis Functions",
    paper_ready=PAPER_READY_PLOTS,
)

plot_basis_functions(
    x=vals,
    Bases=hippo_lag(u),
    ks=range(N),
    data_name="whitenoise",
    domain_name="s",
    basis_type="legendre",
    num_functions_front=None,
    num_functions_back=None,
    title="Evolution of LagHiPPO Coefficients",
    paper_ready=PAPER_READY_PLOTS,
)

plot_reconstructed_signal(
    vals,
    u,
    u_hippo,
    data_name="LagHiPPO",
    domain_name="s",
    title=f"[0,t] ",
    paper_ready=PAPER_READY_PLOTS,
)

plot_signal(
    vals,
    hippo_lag.measure_fn(-hippo_lag.vals),
    "Whitenoise",
    "s",
    function_name=r"$\omega_t$",
    plot_markers=False,
    paper_ready=PAPER_READY_PLOTS,
)

# %%
