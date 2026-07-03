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
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")

import seaborn as sns

sns.set(
    rc={
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "animation.html": "jshtml",
        "animation.embed_limit": 100,  # Max animation size in Mb
    }
)
sns.set_context("notebook")
sns.set_style("whitegrid")  # or 'ticks'

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


# %% Define transition & measure & basis
def transition(measure, N, **measure_args):
    # Laguerre (translated)
    if measure == "lagt":
        b = measure_args.get("beta", 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Legendre (translated)
    elif measure == "legt":
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** 0.5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.0) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # Legendre (scaled)
    elif measure == "legs":
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = (
            B.copy()
        )  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == "fourier":
        freqs = np.arange(N // 2)
        d = np.stack([np.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
        A = 2 * np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2
        B[0] = 2**0.5
        A = A - B[:, None] * B[None, :]
        # A = A - np.eye(N)
        B *= 2**0.5
        B = B[:, None]

    return A, B


def measure(method, c=0.0, sigma_deriv_func=None):
    if method == "legt":
        fn = lambda x: np.heaviside(x, 0.0) * np.heaviside(1.0 - x, 0.0)
    elif method == "legs":
        fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
    elif method == "lagt":
        fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
    elif method in ["fourier"]:
        fn = lambda x: np.heaviside(x, 1.0) * np.heaviside(1.0 - x, 1.0)
    elif method == "custom_lag":
        if sigma_deriv_func is None:
            raise ValueError(
                "`sigma_deriv` function must be provided for 'custom_lag' method."
            )
        # Assumes 'x' is lookback time (tau = t-s) and the present is t=0.
        # This makes absolute time s = -x.
        fn = lambda x: np.heaviside(x, 1.0) * sigma_deriv_func(0, -x)
    else:
        raise NotImplementedError
    fn_tilted = lambda x: np.exp(c * x) * fn(x)
    return fn_tilted


def basis(
    method, N, vals, c=0.0, truncate_measure=True, phi_func=None, sigma_func=None
):
    """
    vals: list of times (forward in time)
    sigma_func (callable): The time-warp function, sigma(t, s).
    sigma_deriv (callable): The derivative of the warp, needed for truncation.

    returns: shape (T, N) where T is length of vals
    """
    if method == "legt":
        eval_matrix = ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1).T
        eval_matrix *= (2 * np.arange(N) + 1) ** 0.5 * (-1) ** np.arange(N)
    elif method == "legs":
        _vals = np.exp(-vals)
        eval_matrix = ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * _vals).T  # (L, N)
        eval_matrix *= (2 * np.arange(N) + 1) ** 0.5 * (-1) ** np.arange(N)
    elif method == "lagt":
        vals = vals[::-1]
        eval_matrix = ss.eval_genlaguerre(np.arange(N)[:, None], 0, vals)
        eval_matrix = eval_matrix * np.exp(-vals / 2)
        eval_matrix = eval_matrix.T
    elif method == "fourier":
        cos = 2**0.5 * np.cos(
            2 * np.pi * np.arange(N // 2)[:, None] * (vals)
        )  # (N/2, T/dt)
        sin = 2**0.5 * np.sin(
            2 * np.pi * np.arange(N // 2)[:, None] * (vals)
        )  # (N/2, T/dt)
        cos[0] /= 2**0.5
        eval_matrix = np.stack([cos.T, sin.T], axis=-1).reshape(-1, N)  # (T/dt, N)

    #     print("eval_matrix shape", eval_matrix.shape)

    if truncate_measure:
        eval_matrix[measure(method)(vals) == 0.0] = 0.0

    p = torch.tensor(eval_matrix)
    p *= np.exp(-c * vals)[:, None]  # [::-1, None]
    return p


# %% HiPPO Scale
class HiPPOScale(nn.Module):
    """Vanilla HiPPO-LegS model (scale invariant instead of time invariant)"""

    def __init__(self, N, method="legs", max_length=1024, discretization="bilinear"):
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
            if discretization == "forward":
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization == "backward":
                A_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At, np.eye(N), lower=True
                )
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, Bt, lower=True)
            elif discretization == "bilinear":
                A_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True
                )
                B_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At / 2, Bt, lower=True
                )
            else:  # ZOH
                A_stacked[t - 1] = la.expm(A * (math.log(t + 1) - math.log(t)))
                B_stacked[t - 1] = la.solve_triangular(
                    A, A_stacked[t - 1] @ B - B, lower=True
                )
        self.register_buffer("A_stacked", torch.Tensor(A_stacked))  # (max_length, N, N)
        self.register_buffer("B_stacked", torch.Tensor(B_stacked))  # (max_length, N)

        vals = np.linspace(0.0, 1.0, max_length)
        self.eval_matrix = torch.Tensor(
            (B[:, None] * ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)).T
        )

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
        u = torch.transpose(u, 0, -2)  # (length, ..., N)

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
    """Linear time invariant x' = Ax + Bu"""

    def __init__(
        self,
        N,
        method="legt",
        dt=1.0,
        T=1.0,
        discretization="bilinear",
        scale=False,
        c=0.0,
    ):
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
        A = A + np.eye(N) * c
        self.A = A
        self.B = B.squeeze(-1)
        self.measure_fn = measure(method)

        C = np.ones((1, N))
        D = np.zeros((1,))
        dA, dB, _, _, _ = signal.cont2discrete(
            (A, B, C, D), dt=dt, method=discretization
        )

        dB = dB.squeeze(-1)

        self.register_buffer("dA", torch.Tensor(dA))  # (N, N)
        self.register_buffer("dB", torch.Tensor(dB))  # (N,)

        self.vals = np.arange(0.0, T, dt)
        self.eval_matrix = basis(self.method, self.N, self.vals, c=self.c)  # (T/dt, N)
        self.measure = measure(self.method)(self.vals)

    def forward(self, inputs, fast=True, verbose=False):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """
        start_time = time.time()
        inputs = inputs.unsqueeze(-1)
        u = inputs * self.dB  # (length, ..., N)

        if fast:
            dA = repeat(self.dA, "m n -> l m n", l=u.size(0))
            result = variable_unroll_matrix(dA, u)
            end_time = time.time()
            if verbose:
                print(
                    f"HiPPO(time-invariant) Total computation time for fast forward pass: {end_time - start_time:.2f} seconds"
                )
            return result

        c = torch.zeros(u.shape[1:]).to(inputs)
        cs = []
        time_start = time.time()
        for f in inputs:
            c = F.linear(c, self.dA) + self.dB * f
            cs.append(c)
        time_end = time.time()
        if verbose:
            print(
                f"HiPPO(time-invariant) Total computation time for forward pass: {time_end - time_start:.2f} seconds"
            )
        return torch.stack(cs, dim=0)

    def reconstruct(
        self, c, evals=None
    ):  # TODO take in a times array for reconstruction
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
