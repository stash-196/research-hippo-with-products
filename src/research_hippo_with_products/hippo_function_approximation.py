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

# %%
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


# %%
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

def measure(method, c=0.0):
    if method == 'legt':
        fn = lambda x: np.heaviside(x, 0.0) * np.heaviside(1.0-x, 0.0)
    elif method == 'legs':
        fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
    elif method == 'lagt':
        fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
    elif method in ['fourier']:
        fn = lambda x: np.heaviside(x, 1.0) * np.heaviside(1.0-x, 1.0)
    else: raise NotImplementedError
    fn_tilted = lambda x: np.exp(c*x) * fn(x)
    return fn_tilted



# %%
def basis(method, N, vals, c=0.0, truncate_measure=True):
    """
    vals: list of times (forward in time)
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


# %%
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


# %%
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

        inputs = inputs.unsqueeze(-1)
        u = inputs * self.dB # (length, ..., N)

        if fast:
            dA = repeat(self.dA, 'm n -> l m n', l=u.size(0))
            return variable_unroll_matrix(dA, u)

        c = torch.zeros(u.shape[1:]).to(inputs)
        cs = []
        for f in inputs:
            c = F.linear(c, self.dA) + self.dB * f
            cs.append(c)
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


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
from tqdm import tqdm
from scipy.special import legendre
import time

def phi(n, z):
    y = 2 * z - 1
    return np.sqrt(2 * n + 1) * legendre(n)(y)

# Warp functions for HiPPO-LegS
def sigma(t, s):
    return np.exp(s - t)

def sigma_inv(t, z):
    return t + np.log(z)

def get_gauss_quadrature(order, a=0, b=1):
    """Get Gauss-Legendre nodes and weights on interval [a, b]."""
    x, w = roots_legendre(order)  # On [-1, 1]
    z = a + (b - a) * (x + 1) / 2
    w = w * (b - a) / 2
    return z, w

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
    legendre_evals = eval_legendre(n_values[np.newaxis, :], y[:, np.newaxis])
    normalization = np.sqrt(2 * n_values + 1)
    return normalization * legendre_evals

# Lag operator transition matrices
def lag_transition(N, dt=1.0, quad_order=None):
    if quad_order is None:
        quad_order = 2 * N  # Exact for degree < 2N (safe for Legendre products up to degree 2N-2)

    # Direct computation of A_infty via quadrature on [0,1]
    z_nodes_a, z_weights_a = get_gauss_quadrature(quad_order, 0, 1)
    n_vals = np.arange(N)
    eval_n = vectorized_phi(n_vals, z_nodes_a)  # (quad_order, N)
    warped_z = np.exp(-dt) * z_nodes_a
    eval_m = vectorized_phi(n_vals, warped_z)  # (quad_order, N)



    # Fix: Sum over quadrature points to get (N, N)
    A_infty_vec = np.einsum('i,ik,il->kl', z_weights_a, eval_n, eval_m)
    start_time = time.time()
    print(f"\nA_infty computed via vectorized Gauss quadrature (order {quad_order}):")
    print(np.round(A_infty_vec, 3))
    end_time = time.time()
    print(f"Total computation time for A_infty: {end_time - start_time:.2f} seconds")

    # Compute B via quadrature on [exp(-dt), 1]
    z_min = np.exp(-dt)
    z_max = 1.0
    z_nodes_b, z_weights_b = get_gauss_quadrature(quad_order, z_min, z_max)
    basis_evals_b = vectorized_phi(n_vals, z_nodes_b)  # (quad_order, N)

    # B = basis_evals_b.T @ z_weights_b (N, quad_order) @ (quad_order,) -> (N,)
    B_zoh_vec = basis_evals_b.T @ z_weights_b

    start_time = time.time()
    print(f"\nB computed via vectorized Gauss quadrature:")
    print(np.round(B_zoh_vec, 3))
    end_time = time.time()
    print(f"Total computation time for B: {end_time - start_time:.2f} seconds")

    return A_infty_vec, B_zoh_vec

class LagHiPPO(nn.Module):
    """ Lag Operator-based time invariant x_{t+1} = A x_t + B u_t """
    def __init__(self, N, dt=1.0, T=1.0, c=0.0):
        """
        N: the order of the HiPPO projection
        dt: discretization step size
        T: total time for evaluation matrix
        c: optional decay parameter
        """
        super().__init__()
        self.N = N
        self.dt = dt
        self.T = T
        self.c = c

        # Compute lag-based transition matrices
        dA, dB = lag_transition(N, dt=self.dt)

        self.register_buffer('dA', torch.Tensor(dA))  # (N, N)
        self.register_buffer('dB', torch.Tensor(dB))  # (N,)

        # Evaluation matrix for reconstruction
        self.vals = np.arange(0.0, T, dt)
        self.eval_matrix = self._compute_eval_matrix()  # (len(vals), N)

        # Measure for truncation (exponential decay)
        self.measure = self._compute_measure()

    def _compute_eval_matrix(self):
        z_warped = np.exp(-self.vals)
        eval_matrix = np.zeros((len(self.vals), self.N))
        for k in range(len(self.vals)):
            for n in range(self.N):
                eval_matrix[k, n] = phi(n, z_warped[k]) * np.exp(-self.c * self.vals[k])
        return torch.Tensor(eval_matrix)

    def _compute_measure(self):
        return np.exp(-self.vals)

    def forward(self, inputs, fast=True):
        """
        inputs : (length, ...)
        output : (length, ..., N)
        """
        inputs = inputs.unsqueeze(-1)
        u = inputs * self.dB  # (length, ..., N)

        if fast:
            dA = self.dA.unsqueeze(0).repeat(u.size(0), 1, 1)  # (L, N, N)
            return variable_unroll_matrix(dA, u)

        c = torch.zeros(u.shape[1:], device=inputs.device)
        cs = []
        for f in inputs:
            c = torch.matmul(self.dA, c.unsqueeze(-1)).squeeze(-1) + self.dB * f
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c, evals=None):
        """
        c: (..., N,) HiPPO coefficients
        output: (..., L,)
        """
        if evals is not None:
            z_warped = np.exp(-evals)
            eval_matrix = np.zeros((len(evals), self.N))
            for k in range(len(evals)):
                for n in range(self.N):
                    eval_matrix[k, n] = phi(n, z_warped[k]) * np.exp(-self.c * evals[k])
            eval_matrix = torch.Tensor(eval_matrix)
        else:
            eval_matrix = self.eval_matrix

        if self.measure is not None:
            m = self.measure[self.measure != 0.0]
            eval_matrix = eval_matrix[:len(m), :]

        c = c.unsqueeze(-1)
        y = torch.matmul(eval_matrix.to(c.device), c)
        return y.squeeze(-1).flip(-1)
# %%
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

# %%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
np.random.seed(0)
def plot(T, dt, N, freq):
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

    for method in lti_methods:
        hippo = HiPPO(method=method, N=N, dt=dt, T=T).to(device)
        u_hippo = hippo.reconstruct(hippo(u))[-1].cpu()
        plt.plot(vals[-len(u_hippo):], u_hippo, label=method)

    # Original HiPPO-LegS, which uses time-varying SSM x' = 1/t [ Ax + Bu]
    # we call this "linear scale invariant"
    lsi_methods = ['legs']
    for method in lsi_methods:
        hippo = HiPPOScale(N=N, method=method, max_length=int(T/dt)).to(device)
        u_hippo = hippo.reconstruct(hippo(u))[-1].cpu()
        plt.plot(vals[-len(u_hippo):], u_hippo, label=method+' (scaled)')

    # Lag operator-based HiPPO-LagS
    hippo = LagHiPPO(N=N, dt=dt, T=T).to(device)
    u_hippo = hippo.reconstruct(hippo(u))[-1].cpu()
    plt.plot(vals[-len(u_hippo):], u_hippo, label='lag operator')

    # plt.xlabel('Time (normalized)', labelpad=-10)
    # plt.savefig(f'function_approximation.pdf', bbox_inches='tight')
    plt.legend()
    plt.show()
    plt.close()


plot(T=3, dt=1e-3, N=64, freq=3.0)

# %%
# Animation code from "How to Train Your HiPPO"

sns.set_style('ticks')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def plt_lines(x, y, color, size, label=None):
    return plt.plot(x, y, color, linewidth=size, label=label)[0]

def update_lines(ln, x, y):
    ln.set_data(x, y)

def animate_hippo(
    method,
    T=5, dt=5e-4, N=64, freq=20.0,
    interval=100,
    plot_hippo=False, hippo_offset=0.0, label_hippo=False,
    plot_measure=False, measure_offset=-3.0, label_measure=False,
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
    u = u + torch.FloatTensor(np.sin(1.5*np.pi/T*np.arange(0, T+dt, dt))) # add 3/4 of a sin cycle
    u = u.to(device)

    hippo = HiPPO(method=method, N=N, dt=dt, T=T).to(device)
    coef_hippo = hippo(u).cpu().numpy()
    h_hippo = hippo.reconstruct(hippo(u)).cpu().numpy()
    u = u.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 4))

    if animate_u:
        ln_u = plt_lines([], [], 'k', size, label='Input $u(t)$')
    else:
        plt_lines(vals, u, 'k', size, label='Input $u(t)$')

    if plot_hippo:
        label_args = {'label': 'HiPPO reconstruction'} if label_hippo else {}
        ln = plt_lines([], [], size=size, color='red', **label_args)

    if plot_measure:
        label_args = {'label': 'HiPPO Measure'} if label_measure else {}
        ln_measure = plt_lines(vals, np.zeros(len(vals))+measure_offset, size=size, color='green', **label_args)

    if plot_coeff is None: plot_coeff = []
    if isinstance(plot_coeff, int): plot_coeff = [plot_coeff]
    if len(plot_coeff) > 0:
        ln_coeffs = [
            plt_lines([], [], size=size, color='blue')
            for _ in plot_coeff
        ]
        plt_lines([], [], size=size, color='blue', label='State $x(t)$') # For the legend


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
    s4 = np.sum(coef_hippo * C, axis=-1)
    max_s4 = 0.0
    if plot_s4:
        ln_s4 = plt_lines([], [], size=size, color='red', label='Output $y(t)$')
        max_s4 = np.max(s4)+s4_offset

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
        m[:frame] = hippo.measure_fn(np.arange(frame)*dt)[::-1]
        xdata = vals
        if plot_measure:
            update_lines(ln_measure, xdata, m+measure_offset)

        if plot_hippo:
            ydata = h_hippo[frame] + hippo_offset
            m2 = hippo.measure_fn(np.arange(len(ydata))*dt)[::-1]
            # Remove reconstruction where measure is 0
            ydata[m2 == 0.0] = np.nan
            xdata = np.arange(frame-len(ydata), frame)
            update_lines(ln, xdata, ydata)

        if len(plot_coeff) > 0:
            for coeff, ln_coeff in zip(plot_coeff, ln_coeffs):
                update_lines(ln_coeff, np.arange(frame), coef_hippo[:frame, coeff] + coeff_offset)
        if plot_s4: # Only scale case; scale case should copy plot_hippo logic
            update_lines(ln_s4, np.arange(0, frame), s4[:frame] + s4_offset)

        if plot_vline:
            ln_vline.set_xdata([frame, frame])

        return []

    ani = FuncAnimation(fig, update,
                        frames=np.arange(0, int(T*1000/interval)+1)*int(interval/1000/dt),
                        interval=interval,
                        init_func=init, blit=True)

    return ani

# %% [markdown]
# Here, an input signal $u(t)$ (Black) is processed by the HiPPO operator $x'(t) = \boldsymbol{A}x(t) + \boldsymbol{B}u(t)$
#  for 10000 steps, maintaining a state $x(t) \in \mathbb{R}^{64}$. At all times, the current state represents a compression of the history of $u(t)$ and can be linearly projected to approximately reconstruct it (Red). This approximation is optimal with respect to an exponentially-decaying measure (Green).

# %%
# Visualize HiPPO online reconstruction

ani = animate_hippo(
    'legs', # Try 'legt' or 'fourier'
    T=5, dt=5e-4, N=64, interval=100,
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

ani = animate_hippo(
    'legs', # Try 'legt' or 'fourier'
    T=5, dt=5e-4, N=64, interval=100,
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



