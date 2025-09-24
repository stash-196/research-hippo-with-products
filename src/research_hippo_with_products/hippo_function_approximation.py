# %%
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from scipy import signal
from scipy import linalg as la
from scipy import special as ss
from scipy.special import roots_legendre, eval_legendre
from einops import rearrange, repeat, reduce
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Use 'notebook' instead for interactive plots
%matplotlib inline
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')

import seaborn as sns
sns.set(rc={
    "figure.dpi":300,
    'savefig.dpi':300,
    'animation.html':'jshtml',
    'animation.embed_limit':100, # Max animation size in Mb
})
sns.set_context('notebook')
sns.set_style('whitegrid') # or 'ticks'

# %% Define useful functions
# Utility functions for performing matrix scans faster
# The HiPPO functions also have pedagogical sequential versions, so this cell is not required

def shift_up(a, s=None, drop=True, dim=0):
    assert dim == 0
    if s is None:
        s = torch.zeros_like(a[0, ...])
    s = s.unsqueeze(dim)
    if drop:
        a = a[:-1, ...]
    return torch.cat((s, a), dim=dim)

def batch_mult(A, u, has_batch=None):
    """ Matrix mult A @ u with special case to save memory if u has additional batch dim

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
    v = (A @ u)
    if has_batch:
        v = v.permute([0] + [len(u.shape)-1] + list(range(1, len(u.shape)-1)))
    else:
        v = v[..., 0]
    return v

def interleave(a, b, uneven=False, dim=0):
    """ Interleave two tensors of same shape """
    # assert(a.shape == b.shape)
    assert dim == 0 # TODO temporary to make handling uneven case easier
    if dim < 0:
        dim = N + dim
    if uneven:
        a_ = a[-1:, ...]
        a = a[:-1, ...]
    c = torch.stack((a, b), dim+1)
    out_shape = list(a.shape)
    out_shape[dim] *= 2
    c = c.view(out_shape)
    if uneven:
        c = torch.cat((c, a_), dim=dim)
    return c


def variable_unroll_general_sequential(A, u, s, op, variable=True):
    """ Unroll with variable (in time/length) transitions A with general associative operation

    A : ([L], ..., N, N) dimension L should exist iff variable is True
    u : (L, [B], ..., N) updates
    s : ([B], ..., N) start state
    output : x (..., N)
    x[i, ...] = A[i]..A[0] s + A[i..1] u[0] + ... + A[i] u[i-1] + u[i]
    """

    if not variable:
        A = A.expand((u.shape[0],) + A.shape)

    outputs = []
    for (A_, u_) in zip(torch.unbind(A, dim=0), torch.unbind(u, dim=0)):
        s = op(A_, s)
        s = s + u_
        outputs.append(s)

    output = torch.stack(outputs, dim=0)
    return output

def variable_unroll_general(A, u, s, op, compose_op=None, sequential_op=None, variable=True, recurse_limit=16):
    """ Bottom-up divide-and-conquer version of variable_unroll.

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

    u_10 = op(A_1, u_0_) # batch_mult(A_1, u_0_, has_batch)
    u_10 = u_10 + u_1
    A_10 = compose_op(A_1, A_0_)

    # Recursive call
    x_1 = variable_unroll_general(A_10, u_10, s, op, compose_op, sequential_op, variable=variable, recurse_limit=recurse_limit)

    x_0 = shift_up(x_1, s, drop=not uneven)
    x_0 = op(A_0, x_0) # batch_mult(A_0, x_0, has_batch)
    x_0 = x_0 + u_0


    x = interleave(x_0, x_1, uneven, dim=0) # For some reason this interleave is slower than in the (non-variable) unroll_recursive
    return x

def variable_unroll_matrix(A, u, s=None, variable=True, recurse_limit=16):
    if s is None:
        s = torch.zeros_like(u[0])
    has_batch = len(u.shape) >= len(A.shape)
    op = lambda x, y: batch_mult(x, y, has_batch)
    sequential_op = lambda x, y: batch_mult(x.unsqueeze(0), y.unsqueeze(0), has_batch)[0]
    matmul = lambda x, y: x @ y
    return variable_unroll_general(A, u, s, op, compose_op=matmul, sequential_op=sequential_op, variable=variable, recurse_limit=recurse_limit)


# %% Define transition & measure & basis
def transition(measure, N, **measure_args):
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == 'fourier':
        freqs = np.arange(N//2)
        d = np.stack([np.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = 2*np.pi*(-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2
        B[0] = 2**.5
        A = A - B[:, None] * B[None, :]
        # A = A - np.eye(N)
        B *= 2**.5
        B = B[:, None]

    return A, B

def measure(method, c=0.0, sigma_deriv_func=None):
    if method == 'legt':
        fn = lambda x: np.heaviside(x, 0.0) * np.heaviside(1.0-x, 0.0)
    elif method == 'legs':
        fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
    elif method == 'lagt':
        fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
    elif method in ['fourier']:
        fn = lambda x: np.heaviside(x, 1.0) * np.heaviside(1.0-x, 1.0)
    elif method == 'custom_lag':
        if sigma_deriv_func is None:
            raise ValueError("`sigma_deriv` function must be provided for 'custom_lag' method.")
        # Assumes 'x' is lookback time (tau = t-s) and the present is t=0.
        # This makes absolute time s = -x.
        fn = lambda x: np.heaviside(x, 1.0) * sigma_deriv_func(0, -x)
    else: raise NotImplementedError
    fn_tilted = lambda x: np.exp(c*x) * fn(x)
    return fn_tilted


def basis(method, N, vals, c=0.0, truncate_measure=True, phi_func=None, sigma_func=None):
    """
    vals: list of times (forward in time)
    sigma_func (callable): The time-warp function, sigma(t, s).
    sigma_deriv (callable): The derivative of the warp, needed for truncation.

    returns: shape (T, N) where T is length of vals
    """
    if method == 'legt':
        eval_matrix = ss.eval_legendre(np.arange(N)[:, None], 2*vals-1).T
        eval_matrix *= (2*np.arange(N)+1)**.5 * (-1)**np.arange(N)
    elif method == 'legs':
        _vals = np.exp(-vals)
        eval_matrix = ss.eval_legendre(np.arange(N)[:, None], 1-2*_vals).T # (L, N)
        eval_matrix *= (2*np.arange(N)+1)**.5 * (-1)**np.arange(N)
    elif method == 'lagt':
        vals = vals[::-1]
        eval_matrix = ss.eval_genlaguerre(np.arange(N)[:, None], 0, vals)
        eval_matrix = eval_matrix * np.exp(-vals / 2)
        eval_matrix = eval_matrix.T
    elif method == 'fourier':
        cos = 2**.5 * np.cos(2*np.pi*np.arange(N//2)[:, None]*(vals)) # (N/2, T/dt)
        sin = 2**.5 * np.sin(2*np.pi*np.arange(N//2)[:, None]*(vals)) # (N/2, T/dt)
        cos[0] /= 2**.5
        eval_matrix = np.stack([cos.T, sin.T], axis=-1).reshape(-1, N) # (T/dt, N)

#     print("eval_matrix shape", eval_matrix.shape)


    if truncate_measure:
        eval_matrix[measure(method)(vals) == 0.0] = 0.0

    p = torch.tensor(eval_matrix)
    p *= np.exp(-c*vals)[:, None] # [::-1, None]
    return p


# %% HiPPO Scale
class HiPPOScale(nn.Module):
    """ Vanilla HiPPO-LegS model (scale invariant instead of time invariant) """
    def __init__(self, N, method='legs', max_length=1024, discretization='bilinear'):
        """
        max_length: maximum sequence length
        """
        super().__init__()
        self.N = N
        A, B = transition(method, N)
        B = B.squeeze(-1)
        A_stacked = np.empty((max_length, N, N), dtype=A.dtype)
        B_stacked = np.empty((max_length, N), dtype=B.dtype)
        for t in range(1, max_length + 1):
            At = A / t
            Bt = B / t
            if discretization == 'forward':
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization == 'backward':
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, np.eye(N), lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, Bt, lower=True)
            elif discretization == 'bilinear':
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, Bt, lower=True)
            else: # ZOH
                A_stacked[t - 1] = la.expm(A * (math.log(t + 1) - math.log(t)))
                B_stacked[t - 1] = la.solve_triangular(A, A_stacked[t - 1] @ B - B, lower=True)
        self.register_buffer('A_stacked', torch.Tensor(A_stacked)) # (max_length, N, N)
        self.register_buffer('B_stacked', torch.Tensor(B_stacked)) # (max_length, N)

        vals = np.linspace(0.0, 1.0, max_length)
        self.eval_matrix = torch.Tensor((B[:, None] * ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)).T  )

    def forward(self, inputs, fast=True):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """
        start_time = time.time()
        L = inputs.shape[0]

        inputs = inputs.unsqueeze(-1)
        u = torch.transpose(inputs, 0, -2)
        u = u * self.B_stacked[:L]
        u = torch.transpose(u, 0, -2) # (length, ..., N)

        if fast:
            result = variable_unroll_matrix(self.A_stacked[:L], u)

            return result

        c = torch.zeros(u.shape[1:]).to(inputs)
        cs = []
        for t, f in enumerate(inputs):
            c = F.linear(c, self.A_stacked[t]) + self.B_stacked[t] * f
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        a = self.eval_matrix.to(c) @ c.unsqueeze(-1)
        return a


# %% HiPPO LTI
class HiPPO(nn.Module):
    """ Linear time invariant x' = Ax + Bu """
    def __init__(self, N, method='legt', dt=1.0, T=1.0, discretization='bilinear', scale=False, c=0.0):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.method = method
        self.N = N
        self.dt = dt
        self.T = T
        self.c = c

        A, B = transition(method, N)
        A = A + np.eye(N)*c
        self.A = A
        self.B = B.squeeze(-1)
        self.measure_fn = measure(method)

        C = np.ones((1, N))
        D = np.zeros((1,))
        dA, dB, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        dB = dB.squeeze(-1)

        self.register_buffer('dA', torch.Tensor(dA)) # (N, N)
        self.register_buffer('dB', torch.Tensor(dB)) # (N,)

        self.vals = np.arange(0.0, T, dt)
        self.eval_matrix = basis(self.method, self.N, self.vals, c=self.c) # (T/dt, N)
        self.measure = measure(self.method)(self.vals)

    def forward(self, inputs, fast=True):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """
        start_time = time.time()
        inputs = inputs.unsqueeze(-1)
        u = inputs * self.dB # (length, ..., N)

        if fast:
            dA = repeat(self.dA, 'm n -> l m n', l=u.size(0))
            result = variable_unroll_matrix(dA, u)
            end_time = time.time()
            print(f"HiPPO(time-invariant) Total computation time for fast forward pass: {end_time - start_time:.2f} seconds")
            return result

        c = torch.zeros(u.shape[1:]).to(inputs)
        cs = []
        time_start = time.time()
        for f in inputs:
            c = F.linear(c, self.dA) + self.dB * f
            cs.append(c)
        time_end = time.time()
        print(f"HiPPO(time-invariant) Total computation time for forward pass: {time_end - time_start:.2f} seconds")
        return torch.stack(cs, dim=0)

    def reconstruct(self, c, evals=None): # TODO take in a times array for reconstruction
        """
        c: (..., N,) HiPPO coefficients (same as x(t) in S4 notation)
        output: (..., L,)
        """
        if evals is not None:
            eval_matrix = basis(self.method, self.N, evals)
        else:
            eval_matrix = self.eval_matrix

        m = self.measure[self.measure != 0.0]

        c = c.unsqueeze(-1)
        y = eval_matrix.to(c) @ c
        return y.squeeze(-1).flip(-1)


# %% Lag HiPPO
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
from tqdm import tqdm
from scipy.special import legendre
import time

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
def sigma(t, s, tau=1.):
    return np.exp((s - t) / tau)

def sigma_deriv(t, s, tau=1.):
    return np.exp((s - t) / tau) / tau


def sigma_uniform(t, s):
    return (s - t + 1.0)  # Shift to positive, normalize if needed

def sigma_uniform_deriv(t, s):
    return 1.0
# def sigma_inv(t, z):
#     return t + np.log(z)

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


def lag_transition_infty(N, phi_func, dt=1.0, quad_order=None):
    if quad_order is None:
        quad_order = max(2 * N, 4 * N if N > 20 else 2 * N)  #

    n_vals = np.arange(N)

    # Direct computation of A_infty via quadrature on [0,1]
    z_nodes_a, z_weights_a = get_gauss_quadrature(quad_order, 0, 1)
    eval_n = vectorized_basis(phi_func, n_vals, z_nodes_a)  # (quad_order, N)
    warped_z = np.exp(dt) * z_nodes_a
    eval_m = vectorized_basis(phi_func, n_vals, warped_z)  # (quad_order, N)

    # Fix: Sum over quadrature points to get (N, N)
    A_infty = np.einsum("i,ik,il->kl", z_weights_a, eval_n, eval_m)
    start_time = time.time()
    print(f"\nA_infty computed via vectorized Gauss quadrature (order {quad_order}):")
    print(np.round(A_infty, 3))
    end_time = time.time()
    print(f"Total computation time for A_infty: {end_time - start_time:.2f} seconds")

    # Compute B via phi(1)
    B_delta = np.zeros(N)
    for n in range(N):
        B_delta[n] = phi_func(n, 1)
    return A_infty, B_delta  # Return A_infty for discretization


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
        "i,ik,il->kl", z_weights, phi_vals, z_nodes[:, None] * phi_deriv_vals
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
    """ Lag Operator-based time invariant x_{t+1} = A x_t + B u_t """
    def __init__(self, N, phi_func, phi_deriv_func, sigma_func, sigma_deriv_func, dt=1.0, T=1.0, c=0.0, discretization='bilinear', derive_A_from='gen', mode='scale', window=1.0):
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
        dA, dB, _, _, _ = signal.cont2discrete((A_HiPPO, B, C, D), dt=dt, method=discretization)

        if derive_A_from == 'infty':
            A_infty, _ = lag_transition_infty(N, phi_func, dt=self.dt)
            A_infty_corrected = np.linalg.inv(A_infty.T) * np.exp(-dt) 
            dA = A_infty_corrected
            print("\nUsing A_infty for discretization:")

        self.register_buffer('dA', torch.Tensor(dA))  # (N, N)
        self.register_buffer('dB', torch.Tensor(dB))  # (N,)

        # Evaluation matrix for reconstruction
        self.vals = np.arange(0.0, T, dt)
        self.eval_matrix = self._compute_eval_matrix()  # (len(vals), N)
        self.measure = self._compute_measure()  # Using original HiPPO convention

    def _compute_eval_matrix(self):
        start_time = time.time()
        z_warped = self.sigma_func(0, -self.vals)  # (len(vals),)
        n_vals = np.arange(self.N)
        eval_matrix = vectorized_basis(self.phi_func, n_vals, z_warped)  # (len(vals), N)
        decay_factor = np.exp(-self.c * self.vals)  # (len(vals),)
        eval_matrix = eval_matrix * decay_factor[:, np.newaxis]  # Broadcast decay to
        end_time = time.time()
        print(f"LagHiPPO: Total computation time for basis: {end_time - start_time:.2f} seconds")

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
            print(f"LagHiPPO: Total computation time for fast forward pass: {time_end - time_start:.2f} seconds")
            return result

        c = torch.zeros(u.shape[1:], device=inputs.device)
        cs = []
        for f in inputs:
            c = torch.matmul(self.dA, c.unsqueeze(-1)).squeeze(-1) + self.dB * f
            cs.append(c)
        time_end = time.time()
        print(f"LagHiPPO: Total computation time for forward pass: {time_end - time_start:.2f} seconds")
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
            print(f"LagHiPPO: reconstruct() Total computation time for eval_matrix: {time_end - time_start:.2f} seconds")
        else:
            eval_matrix = self.eval_matrix

        c = c.unsqueeze(-1)
        y = torch.matmul(eval_matrix.to(c.device), c)
        return y.squeeze(-1).flip(-1)


# %% White Noise
### Synthetic data generation

def whitesignal(period, dt, freq, rms=0.5, batch_shape=()):
    """
    Produces output signal of length period / dt, band-limited to frequency freq
    Output shape (*batch_shape, period/dt)
    Adapted from the nengo library
    """

    if freq is not None and freq < 1. / period:
        raise ValueError(f"Make ``{freq=} >= 1. / {period=}`` to produce a non-zero signal",)

    nyquist_cutoff = 0.5 / dt
    if freq > nyquist_cutoff:
        raise ValueError(f"{freq} must not exceed the Nyquist frequency for the given dt ({nyquist_cutoff:0.3f})")

    n_coefficients = int(np.ceil(period / dt / 2.))
    shape = batch_shape + (n_coefficients + 1,)
    sigma = rms * np.sqrt(0.5)
    coefficients = 1j * np.random.normal(0., sigma, size=shape)
    coefficients[..., -1] = 0.
    coefficients += np.random.normal(0., sigma, size=shape)
    coefficients[..., 0] = 0.

    set_to_zero = np.fft.rfftfreq(2 * n_coefficients, d=dt) > freq
    coefficients *= (1-set_to_zero)
    power_correction = np.sqrt(1. - np.sum(set_to_zero, dtype=float) / n_coefficients)
    if power_correction > 0.: coefficients /= power_correction
    coefficients *= np.sqrt(2 * n_coefficients)
    signal = np.fft.irfft(coefficients, axis=-1)
    signal = signal - signal[..., :1]  # Start from 0
    return signal


visualize = True
if visualize:
    x = whitesignal(1., freq=3.0, dt=0.01, batch_shape=(2,))
    print(x.shape)
    plt.plot(np.arange(x.shape[-1]), x[0])
    plt.plot(np.arange(x.shape[-1]), x[1])
    plt.show()



# %% [markdown]
# Visualize the HiPPO function reconstruction theory in the spirit of the original paper. A random function $u(t)$ (black) is generated and processed sequentially by a HiPPO operator $x'(t) = A(t) x(t) + B(t) u(t)$. At the end of the sequence, the final state $x(t)$, which has dimension much smaller than the length of the sequence, can be used (via a linear projection) to approximately reconstruct the entire input.
# %% plot Whitenoise
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
np.random.seed(0)
def plot(T, dt, N, freq, plot_domain='s'):
    vals = np.arange(0.0, T, dt)

    u = whitesignal(T, dt, freq=freq)
    u = torch.tensor(u, dtype=torch.float)
    u = u.to(device)

    plt.figure(figsize=(16, 8))
    offset = 0.0
    plt.plot(vals, u.cpu()+offset, 'k', linewidth=1.0, label='Input u(t)')

    # Linear Time Invariant (LTI) methods x' = Ax + Bu
    lti_methods = [
        'legs',
        'legt',
        'fourier',
    ]

    # for method in lti_methods:
    #     hippo = HiPPO(method=method, N=N, dt=dt, T=T).to(device)
    #     u_hippo = hippo.reconstruct(hippo(u))[-1].cpu()
    #     print("u_hippo shape for method ({}): {}".format(method, u_hippo.shape))
    #     plt.plot(vals[-len(u_hippo):], u_hippo, label=method)

    # # Original HiPPO-LegS, which uses time-varying SSM x' = 1/t [ Ax + Bu]
    # # we call this "linear scale invariant"
    # lsi_methods = ['legs']
    # for method in lsi_methods:
    #     hippo = HiPPOScale(N=N, method=method, max_length=int(T/dt)).to(device)
    #     u_hippo = hippo.reconstruct(hippo(u))[-1].cpu()
    #     plt.plot(vals[-len(u_hippo):], u_hippo, label=method+' (scaled)')



    hippo_lti = HiPPO(method='legs', N=N, dt=dt, T=T).to(device)
    u_hippo = hippo_lti.reconstruct(hippo_lti(u))[-1].cpu()
    print("u_hippo shape for method ({}): {}".format('legs', u_hippo.shape))
    plt.plot(vals[-len(u_hippo):], u_hippo, label='legs')

    # # Lag operator-based HiPPO-LagS
    phi_func = phi
    phi_deriv_func = phi_deriv
    sigma_func = lambda t, s: sigma(t, s, tau=1.0)
    sigma_deriv_func = lambda t, s: sigma_deriv(t, s, tau=1.0)
    hippo_lag = LagHiPPO(N=N, phi_func=phi_func, phi_deriv_func=phi_deriv_func, sigma_func=sigma_func, sigma_deriv_func=sigma_deriv_func, dt=dt, T=T, mode='translate').to(device)
    
    u_hippo = hippo_lag.reconstruct(hippo_lag(u))[-1].cpu()
    print("u_hippo shape for LagHiPPO: {}".format(u_hippo.shape))
    plt.plot(vals[-len(u_hippo):], u_hippo, label='lag operator', linestyle='--')

    A_infty, _ = lag_transition_infty(N, phi_func, dt=dt)
    print("A_infty for LagHiPPO:\n", np.round(A_infty.T, 6))
    print("Eigenvalues of A_infty:\n", np.round(np.abs(np.linalg.eigvals(A_infty.T)), 6))

    A_infty_corrected = np.linalg.inv(A_infty.T) * np.exp(-dt)  # Correct for missing exp(-dt) factor in derivation
    print("A_infty_corrected for LagHiPPO:\n", np.round(A_infty_corrected.T, 6))
    print("Eigenvalues of A_infty_corrected:\n", np.round(np.abs(np.linalg.eigvals(A_infty_corrected.T)), 6))

    print("dA for LagHiPPO:\n", np.round(hippo_lag.dA.cpu().numpy(), 6))
    print("Eigenvalues of dA for LagHiPPO:\n", np.round(np.abs(np.linalg.eigvals(hippo_lag.dA.cpu().numpy())), 6))

    hippo_lag_infty = LagHiPPO(N=N, phi_func=phi_func, phi_deriv_func=phi_deriv_func, sigma_func=sigma_func, sigma_deriv_func=sigma_deriv_func, dt=dt, T=T, mode='translate', derive_A_from='infty').to(device)

    u_hippo = hippo_lag_infty.reconstruct(hippo_lag_infty(u))[-1].cpu()
    print("u_hippo for LagHiPPO:\n{}".format(u_hippo))
    plt.plot(vals[-len(u_hippo):], u_hippo, label=r'lag operator $A_{\infty}$', linestyle='--')

    plt.plot(vals[-len(u_hippo):], hippo_lti.measure[::-1]-2, linewidth=1.0, label='measure lti')
    plt.plot(vals[-len(u_hippo):], hippo_lag.measure[::-1]-2, linewidth=1.0, label='measure lag', linestyle='--')

    # plt.xlabel('Time (normalized)', labelpad=-10)
    # plt.savefig(f'function_approximation.pdf', bbox_inches='tight')
    plt.legend()
    plt.show()
    plt.close()


plot(T=10, dt=1e-3, N=64, freq=3.0)

# %%
# Animation code from "How to Train Your HiPPO"
sns.set_style('ticks')
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

def plt_lines(x, y, color, size, label=None):
    return plt.plot(x, y, color, linewidth=size, label=label)[0]

def update_lines(ln, x, y):
    ln.set_data(x, y)

def animate_hippo(
    model,  # Pass model instance (HiPPO or LagHiPPO)
    T=5, dt=5e-4, N=64, freq=20.0,
    interval=100,
    plot_hippo=True, hippo_offset=0.0, label_hippo=False,
    plot_measure=True, measure_offset=-3.0, label_measure=False,
    plot_coeff=None, coeff_offset=3.0,
    plot_s4=False, s4_offset=6.0,
    plot_hippo_type='line', plot_measure_type='line', plot_coeff_type='line',
    size=1.0,
    plot_legend=True, plot_xticks=True, plot_box=True,
    plot_vline=False,
    animate_u=False,
    seed=2,
):
    np.random.seed(seed)

    vals = np.arange(0, int(T/dt)+1)
    L = int(T/dt)+1

    u = torch.FloatTensor(whitesignal(T, dt, freq=freq))
    u = F.pad(u, (1, 0))
    u = u + torch.FloatTensor(np.sin(1.5*np.pi/T*np.arange(0, T+dt, dt)))  # Add 3/4 of a sin cycle
    u = u.to(device)

    # Use provided model
    coef_model = model(u).cpu().numpy()
    h_model = model.reconstruct(model(u)).cpu().numpy()
    # print('Reconstruction MSE:', np.mean((h_model.flatten() - u.cpu().numpy().flatten())**2))
    print('u shape: ', u.shape)
    print('u: ', u)
    print('coef_model shape: ', coef_model.shape)
    print('coef_model: ', coef_model)
    print('h_model shape: ', h_model.shape)
    print('h_model: ', h_model)
    u = u.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 4))

    if animate_u:
        ln_u = plt_lines([], [], 'k', size, label='Input $u(t)$')
    else:
        plt_lines(vals, u, 'k', size, label='Input $u(t)$')

    if plot_hippo:
        label_args = {'label': f'{model.__class__.__name__} reconstruction'} if label_hippo else {}
        ln = plt_lines([], [], size=size, color='red', **label_args)

    if plot_measure:
        label_args = {'label': f'{model.__class__.__name__} Measure'} if label_measure else {}
        ln_measure = plt_lines(vals, np.zeros(len(vals))+measure_offset, size=size, color='green', **label_args)

    if plot_coeff is None: plot_coeff = []
    if isinstance(plot_coeff, int): plot_coeff = [plot_coeff]
    if len(plot_coeff) > 0:
        ln_coeffs = [plt_lines([], [], size=size, color='blue') for _ in plot_coeff]
        plt_lines([], [], size=size, color='blue', label='State $x(t)$')  # For legend

    ### Y AXIS LIMITS
    if plot_measure:
        min_y = measure_offset
    else:
        min_y = np.min(u)

    if len(plot_coeff) > 0:
        max_u = np.max(u) + coeff_offset
    else:
        max_u = np.max(u)

    C = np.random.random(N)
    s4 = np.sum(coef_model * C, axis=-1)
    max_s4 = 0.0
    if plot_s4:
        ln_s4 = plt_lines([], [], size=size, color='red', label='Output $y(t)$')
        max_s4 = np.max(s4) + s4_offset

    if plot_vline:
        ln_vline = ax.axvline(0, ls='-', color='k', lw=1)

    if plot_legend:
        plt.legend(loc='upper left', fontsize='x-small')

    def init():
        left_endpoint = vals[0]
        ax.set_xlim(left_endpoint, vals[-1]+1)
        ax.set_ylim(min_y, max(max_u, max_s4))
        ax.set_yticks([])
        if not plot_xticks: ax.set_xticks([])
        if not plot_box: plt.box(False)
        return [] # ln,

    def update(frame):
        if animate_u:
            xdata = np.arange(frame)
            ydata = u[:frame]
            update_lines(ln_u, xdata, ydata)

        m = np.zeros(len(vals))
        m[:frame] = model.measure_fn(np.arange(frame) * dt)[::-1] if hasattr(model, 'measure_fn') else model.measure[:frame][::-1]
        xdata = vals
        if plot_measure:
            update_lines(ln_measure, xdata, m + measure_offset)

        if plot_hippo:
            ydata = h_model[frame] + hippo_offset
            m2 = model.measure_fn(np.arange(len(ydata)) * dt)[::-1] if hasattr(model, 'measure_fn') else model.measure[:len(ydata)][::-1]
            ydata[m2 == 0.0] = np.nan
            xdata = np.arange(frame - len(ydata), frame)
            update_lines(ln, xdata, ydata)

        if len(plot_coeff) > 0:
            for coeff, ln_coeff in zip(plot_coeff, ln_coeffs):
                update_lines(ln_coeff, np.arange(frame), coef_model[:frame, coeff] + coeff_offset)
        if plot_s4:
            update_lines(ln_s4, np.arange(0, frame), s4[:frame] + s4_offset)

        if plot_vline:
            ln_vline.set_xdata([frame, frame])

        return []

    ani = FuncAnimation(fig, update,
                        frames=np.arange(0, int(T*1000/interval)+1) * int(interval/1000/dt),
                        interval=interval,
                        init_func=init, blit=True)

    return ani
#%%


# %% [markdown]
# Here, an input signal $u(t)$ (Black) is processed by the HiPPO operator $x'(t) = \boldsymbol{A}x(t) + \boldsymbol{B}u(t)$
#  for 10000 steps, maintaining a state $x(t) \in \mathbb{R}^{64}$. At all times, the current state represents a compression of the history of $u(t)$ and can be linearly projected to approximately reconstruct it (Red). This approximation is optimal with respect to an exponentially-decaying measure (Green).

# %%
# Visualize HiPPO online reconstruction
N = 64
T = 5
dt = 5e-4
hippo_lti = HiPPO(method='legs', N=N, dt=dt, T=T).to(device)
ani = animate_hippo(
    # 'legs', # Try 'legt' or 'fourier'
    hippo_lti,
    T=T, dt=dt, N=N, interval=100,
#     T=1, dt=1e-3, N=64, interval=200, # Faster rendering for testing
    size=1.0,

    animate_u=True,
    plot_hippo=True, hippo_offset=0.0, label_hippo=True,
    plot_s4=False, s4_offset=6.0,
    plot_measure=True, measure_offset=-3.0, label_measure=True,
    plot_coeff=[], coeff_offset=3.0,
    plot_legend=True, plot_xticks=True, plot_box=True,
    plot_vline=True,
)
ani


#%%
N = 64
T = 5
dt = 5e-4
hippo_lag = LagHiPPO(N=N, phi_func=phi, phi_deriv_func=phi_deriv, sigma_func=sigma, sigma_deriv_func=sigma_deriv, dt=dt, T=T).to(device)
# hippo_lti = HiPPO(method='legs', N=N, dt=dt, T=T).to(device)
# hippo_lag.dA = hippo_lti.dA
ani = animate_hippo(
    # 'legs', # Try 'legt' or 'fourier'
    hippo_lag,
    T=T, dt=dt, N=N, interval=100,
#     T=1, dt=1e-3, N=64, interval=200, # Faster rendering for testing
    size=1.0,

    animate_u=True,
    plot_hippo=True, hippo_offset=0.0, label_hippo=True,
    plot_s4=False, s4_offset=6.0,
    plot_measure=True, measure_offset=-3.0, label_measure=True,
    plot_coeff=[], coeff_offset=3.0,
    plot_legend=True, plot_xticks=True, plot_box=True,
    plot_vline=True,
)
ani
# %%
# Visualize S4

N = 64
T = 5
dt = 5e-4
hippo_lti = HiPPO(method='legs', N=N, dt=dt, T=T).to(device)
ani = animate_hippo(
    hippo_lti,
    T=T, dt=dt, N=N, interval=100,
    size=1.0,

    animate_u=True,
    plot_hippo=False, hippo_offset=0.0, label_hippo=True,
    plot_s4=True, s4_offset=6.0,
    plot_measure=False, measure_offset=-3.0, label_measure=True,
    plot_coeff=[0,1,2,3], coeff_offset=3.0,
    plot_legend=True, plot_xticks=True, plot_box=True,
    plot_vline=True,
)
ani

#%%


# %%
ani.save(
    's4_operator_legs.gif',
    savefig_kwargs={
        # 'transparent': True,
        # 'bbox_inches': 'tight',
        'pad_inches': 0,
    }
)

# %%



# --- 1. Define the new Linear Warp functions ---

# def linear_warp(t, s, T=1.0):
#     """Maps absolute time [t-T, t] to canonical time [0, 1]."""
#     # In lookback coordinates (s = -vals), this is just vals/T
#     x = (s - (t - T)) / T
#     return (s - (t - T)) / T

# def linear_warp_deriv(t, s, T=1.0):
#     """Derivative w.r.t. s is constant."""
#     x = 1.0 / T
#     return 1.0 / T * np.heaviside(x, 1.0) * np.heaviside(1.0-x, 1.0)

def linear_warp(t, s):
    """Maps absolute time [t-T, t] to canonical time [0, 1]."""
    # In lookback coordinates (s = -vals), this is just vals/T
    return (s - t)

def linear_warp_deriv(t, s):
    """Derivative w.r.t. s is constant."""
    return 1.0

# --- 2. Define a periodic test signal ---

def periodic_signal(vals):
    """A sum of sine waves, perfect for testing the Fourier basis."""
    return (np.sin(2 * np.pi * 2 * vals) +
            0.5 * np.sin(2 * np.pi * 5 * vals))

# --- 3. A new plotting function for your experiment ---

def plot_basis_comparison(T, dt, N):
    vals = np.arange(0.0, T, dt)
    u = torch.tensor(periodic_signal(vals / T), dtype=torch.float) # Normalize time for signal

    plt.figure(figsize=(14, 8))
    plt.plot(vals, u.cpu(), 'k', linewidth=2, label='Input Periodic Signal')

    # --- Model 0: Legendre Basis + Exponential Warp ---
    print("\n--- Testing Legendre + Exponential Warp ---")
    model_fourier = LagHiPPO(
        N=N,
        phi_func=phi,
        phi_deriv_func=phi_deriv,
        sigma_func=sigma,
        sigma_deriv_func=sigma_deriv,
        dt=dt, T=T
    )
    c_fourier = model_fourier(u)
    recon_fourier = model_fourier.reconstruct(c_fourier[-1]).cpu().detach().numpy()
    mse_fourier = np.mean((u.cpu().numpy() - recon_fourier)**2)
    plt.plot(vals, recon_fourier, '--', label=f'Legendre Basis w Exp Warp | MSE: {mse_fourier:.5f}')


    # --- Model 1: Legendre Basis + Linear Warp (HiPPO-LegT) ---
    print("--- Testing Legendre + Linear Warp ---")
    model_legt = LagHiPPO(
        N=N,
        phi_func=phi,
        phi_deriv_func=phi_deriv,
        sigma_func=lambda t, s: linear_warp(t, s),
        sigma_deriv_func=lambda t, s: linear_warp_deriv(t, s),
        dt=dt, T=T,
        mode='translate'
    )
    c_legt = model_legt(u)
    recon_legt = model_legt.reconstruct(c_legt[-1]).cpu().detach().numpy()
    mse_legt = np.mean((u.cpu().numpy() - recon_legt)**2)
    plt.plot(vals, recon_legt, '--', label=f'Legendre Basis | MSE: {mse_legt:.5f}')

    # --- Model 2: Fourier Basis + Linear Warp (Novel Model) ---
    print("\n--- Testing Fourier + Linear Warp ---")
    model_fourier = LagHiPPO(
        N=N,
        phi_func=phi_fourier,
        phi_deriv_func=phi_fourier_deriv,
        sigma_func=lambda t, s: linear_warp(t, s),
        sigma_deriv_func=lambda t, s: linear_warp_deriv(t, s),
        dt=dt, T=T
    )
    c_fourier = model_fourier(u)
    recon_fourier = model_fourier.reconstruct(c_fourier[-1]).cpu().detach().numpy()
    mse_fourier = np.mean((u.cpu().numpy() - recon_fourier)**2)
    plt.plot(vals, recon_fourier, '--', label=f'Fourier Basis | MSE: {mse_fourier:.5f}')



    plt.title(f"Basis Comparison on Periodic Signal (N={N})")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Run the experiment ---
plot_basis_comparison(T=3.0, dt=1e-3, N=32)
# %%
