# Adapted from https://github.com/state-spaces/s4 (Gu et al., 2021, MIT License)
# Implements Lag Operator HiPPO, recovering HiPPO-LegS as a special case.
import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from scipy.special import roots_legendre, eval_legendre
from scipy.linalg import expm
import time
from lag_op_hippo.hippo_by_original_authors import (
    variable_unroll_matrix,
)


# Warp functions for HiPPO-LegS
def sigma(t, s):
    """Exponential warping function."""
    return np.exp((s - t))


def sigma_deriv(t, s):
    """Derivative of exponential warping."""
    return np.exp((s - t))


def sigma_inv(t, z):
    """Inverse of exponential warping: s = t + log(z)."""
    z = np.asarray(z, dtype=np.float64)
    z = np.maximum(z, 1e-300)
    return np.asarray(t, dtype=np.float64) + np.log(z)


def sigma_inv_deriv(t, z):
    """Derivative of inverse warping: ds/dz = 1/z."""
    del t
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / np.maximum(z, 1e-300)


def phi(n, z):
    """Legendre basis function."""
    y = 2 * z - 1
    return np.sqrt(2 * n + 1) * eval_legendre(n, y)


def phi_deriv(n, z):
    """Derivative of Legendre basis function."""
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


def get_composite_gauss_quadrature(order, breakpoints, a=0.0, b=1.0, min_order=8):
    """Composite Gauss-Legendre on [a, b] split at ``breakpoints``.

    Used for piecewise warps (e.g. FouP), whose integrands have kinks at the
    window edges; plain Gauss over [0, 1] converges only algebraically there.
    ``order`` is the total node budget, spread evenly over the segments.
    """
    pts = np.unique(np.clip(np.concatenate([[a, b], np.asarray(breakpoints, float)]), a, b))
    nseg = len(pts) - 1
    per = max(int(np.ceil(order / nseg)), min_order)
    zs, ws = [], []
    for lo, hi in zip(pts[:-1], pts[1:]):
        if hi - lo <= 0:
            continue
        z, w = get_gauss_quadrature(per, lo, hi)
        zs.append(z); ws.append(w)
    return np.concatenate(zs), np.concatenate(ws)


def _quadrature(order, z_breakpoints, a=0.0, b=1.0):
    if z_breakpoints is None:
        return get_gauss_quadrature(order, a, b)
    return get_composite_gauss_quadrature(order, z_breakpoints, a, b)


def vectorized_basis(basis_func, n_values, z_values):
    """Vectorized evaluation of basis function."""
    k = len(z_values)
    N = len(n_values)
    vals = np.zeros((k, N))
    for j, n in enumerate(n_values):
        vals[:, j] = basis_func(n, z_values)
    return vals


def compute_A_infty(
    N, phi_func, sigma_func, sigma_inv_func, dt=1.0, quad_order=None, verbose=False,
    z_breakpoints=None,
):
    """Compute A_infty using Gauss quadrature (composite if ``z_breakpoints``)."""
    if quad_order is None:
        quad_order = max(2 * N, 4 * N if N > 20 else 2 * N)
    n_vals = np.arange(N)
    start_time = time.time()
    z_nodes_a, z_weights_a = _quadrature(quad_order, z_breakpoints)
    eval_n = vectorized_basis(phi_func, n_vals, z_nodes_a)
    # Composition σ_0 ∘ σ_Δ^{-1}: use the *provided* warp, not the module-level LegS σ.
    warped_z = sigma_func(0, sigma_inv_func(dt, z_nodes_a))
    eval_m = vectorized_basis(phi_func, n_vals, warped_z)
    A_infty = np.einsum("i,ik,il->kl", z_weights_a, eval_n, eval_m)
    end_time = time.time()
    if verbose:
        print(np.round(A_infty, 3))
        print(
            f"Total computation time for A_infty: {end_time - start_time:.2f} seconds"
        )
    return A_infty


def compute_A_gen(
    N,
    phi_func,
    phi_deriv_func,
    dt=1.0,
    quad_order=None,
    verbose=False,
    sigma_inv_deriv_func=None,
    z_breakpoints=None,
):
    """Compute A_gen using Gauss quadrature.

    ``z_breakpoints`` (optional) switches to composite Gauss split at the warp's
    kinks (FouP window edges alpha^n); default None keeps the original nodes.

    ``sigma_inv_deriv_func`` is the warp's d(sigma^{-1})/dz. It defaults to the
    LegS (exponential) warp for backward compatibility; LagHiPPO passes the
    model's own warp (fix 2026-11-19: previously the LegS warp was always used,
    which biased A_gen for every non-exponential warp).
    """
    if sigma_inv_deriv_func is None:
        sigma_inv_deriv_func = sigma_inv_deriv
    if quad_order is None:
        quad_order = max(2 * N, 4 * N if N > 20 else 2 * N)
    n_vals = np.arange(N)
    start_time = time.time()
    z_nodes, z_weights = _quadrature(quad_order, z_breakpoints)
    phi_vals = vectorized_basis(phi_func, n_vals, z_nodes)
    phi_deriv_vals = vectorized_basis(phi_deriv_func, n_vals, z_nodes)
    lag_deriv_factor = 1 / np.asarray(sigma_inv_deriv_func(0, z_nodes), dtype=np.float64)
    A_gen = np.einsum(
        "i,ik,il->kl", z_weights, phi_vals, lag_deriv_factor[:, None] * phi_deriv_vals
    )
    end_time = time.time()
    if verbose:
        print(np.round(A_gen, 3))
        print(f"Total computation time for A_gen: {end_time - start_time:.2f} seconds")
    return A_gen


def compute_B_zoh(N, phi_func, dt, quad_order=None, verbose=False, sigma_func=None):
    """Compute B_zoh = int_{sigma_0(-dt)}^{1} phi(z) dz (newest input strip) by Gauss quadrature.

    ``sigma_func`` is the model's warp; the strip's lower limit is
    ``sigma_func(0, -dt)``. Default (None) keeps the LegS value ``e^{-dt}``
    (fix 2026-11-19: the e^{-dt} limit was used for every warp).
    """
    if quad_order is None:
        quad_order = 2 * N
    n_vals = np.arange(N)
    start_time = time.time()
    z_min = np.exp(-dt) if sigma_func is None else float(np.asarray(sigma_func(0.0, -dt)))
    z_max = 1.0
    z_nodes_b, z_weights_b = get_gauss_quadrature(quad_order, z_min, z_max)
    basis_evals_b = vectorized_basis(phi_func, n_vals, z_nodes_b)
    B_zoh_vec = basis_evals_b.T @ z_weights_b
    end_time = time.time()
    if verbose:
        print(np.round(B_zoh_vec, 3))
        print(f"Total computation time for B: {end_time - start_time:.2f} seconds")
    return B_zoh_vec


def compute_B_delta(N, phi_func):
    """Compute B_delta using phi(1)."""
    B_delta = np.zeros(N)
    for n in range(N):
        B_delta[n] = phi_func(n, 1)
    return B_delta


def compute_I_1(N, phi_func, dt=1.0, quad_order=None, verbose=False):
    """Compute I_1 using Gauss quadrature."""
    if quad_order is None:
        quad_order = max(2 * N, 4 * N if N > 20 else 2 * N)
    n_vals = np.arange(N)
    start_time = time.time()
    z_min = np.exp(-dt)
    z_nodes, z_weights = get_gauss_quadrature(quad_order, z_min, 1.0)
    basis_evals = vectorized_basis(phi_func, n_vals, z_nodes)
    I_1_vec = np.einsum("ik,i->k", basis_evals, z_weights)
    end_time = time.time()
    if verbose:
        print(np.round(I_1_vec, 3))
        print(f"Total computation time for I_1: {end_time - start_time:.2f} seconds")
    return I_1_vec


def compute_I_g(N, phi_func, dt=1.0, quad_order=None, verbose=False):
    """Compute I_g using Gauss quadrature."""
    if quad_order is None:
        quad_order = max(2 * N, 4 * N if N > 20 else 2 * N)
    n_vals = np.arange(N)
    start_time = time.time()
    z_min = np.exp(-dt)
    z_nodes, z_weights = get_gauss_quadrature(quad_order, z_min, 1.0)
    basis_evals = vectorized_basis(phi_func, n_vals, z_nodes)
    g_vals = np.log(z_nodes)
    I_g_vec = np.einsum("ik,i,i->k", basis_evals, g_vals, z_weights)
    end_time = time.time()
    if verbose:
        print(np.round(I_g_vec, 3))
        print(f"Total computation time for I_g: {end_time - start_time:.2f} seconds")
    return I_g_vec


class LagHiPPO(nn.Module):
    """Lag Operator-based time invariant x_{t+1} = A x_t + B u_t"""

    def __init__(
        self,
        N,
        phi_func,
        phi_deriv_func,
        sigma_func,
        sigma_deriv_func,
        sigma_inv_func,
        sigma_inv_deriv_func,
        dt=1.0,
        T=1.0,
        c=0.0,
        discretization="bilinear",
        derive_A_from="gen",
        mode="scale",
        window=1.0,
        input_hold="delta",
        quad_order=None,
        a_tilt="legs",
        B_vec=None,
        edge_kappa=None,
        z_breakpoints=None,
    ):
        """
        N: the order of the HiPPO projection
        dt: discretization step size
        T: total time for evaluation matrix
        c: optional decay parameter
        mode: 'scale' for exponential/compressive warps, 'translate' for uniform/truncated warps.
        window: For 'translate' mode, the fixed truncation window (normalized to 1 by default to match HiPPO-LegT).
        a_tilt: How to map A_gen to continuous A.
            'legs' maps to -(A_gen + I).T (HiPPO-LegS),
            'translate' maps to A_gen - (kappa/window) phi(1)phi(1)^T (truncated-window
                edge closure; kappa=1 LegT, 2 FouT, 2(1-alpha)/(1+alpha) FouP),
                with B = (kappa/window) phi(1) unless B_vec is given,
            'translate_legacy' maps to -A_gen (pre-2026-11-19 'translate'),
            'raw' uses A_gen directly.
        edge_kappa: kappa for a_tilt='translate' (default 1.0).
        z_breakpoints: optional warp kinks in z for composite quadrature of
            A_gen / A_infty (None = original single-interval Gauss).

        Exact relation behind 'translate' (z=1 present, z=0 oldest edge; h = measure
        density, h = 1/L on the window, G_{h'} = Gram of phi under h'):
            dc/dt = -(A_gen^T + G_{h'}) c + h(1) phi(1) u(t) - h(0) phi(0) u(t-L)
                  = A_gen c + h(1) phi(1) (u(t) - phi(1)^T c)
                            - h(0) phi(0) (u(t-L) - phi(0)^T c)
        (integration by parts). The shorter form
            dc/dt = A_gen c + (1/L)[phi(1) u(t) - phi(0) u(t-L)]
        holds only when A_gen is skew and G_{h'} = 0 (FouT); it is NOT exact for
        LegT (finite-difference error ~8). Closures of the edge sample:
        LegT u(t-L) ~ phi(0)^T c                 -> A = A_gen - (1/L) phi(1)phi(1)^T, B = phi(1)/L;
        FouT u(t-L) ~ 2 phi(0)^T c - u(t) (mid-point, phi(0)=phi(1))
                                                 -> A = A_gen - (2/L) phi(1)phi(1)^T, B = 2 phi(1)/L.
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
        self.sigma_inv_func = sigma_inv_func
        self.sigma_inv_deriv_func = sigma_inv_deriv_func
        self.mode = mode
        self.window = window
        self.quad_order = quad_order
        # Provenance of discrete (dA, dB) for plots / diagnostics.
        self.derive_A_from = derive_A_from
        self.input_hold = input_hold
        self.a_tilt = a_tilt
        self.discretization = discretization

        # Compute lag-based transition matrices
        # Use A_gen and B_delta + discretization as default (this is analytically equivalent to HiPPO-LegS)
        A_gen = compute_A_gen(
            N,
            self.phi_func,
            self.phi_deriv_func,
            dt=self.dt,
            quad_order=self.quad_order,
            sigma_inv_deriv_func=self.sigma_inv_deriv_func,
            z_breakpoints=z_breakpoints,
        )
        self.A_gen = A_gen
        phi_1 = compute_B_delta(N, self.phi_func)
        self.edge_kappa = 1.0 if edge_kappa is None else float(edge_kappa)
        if a_tilt == "legs":
            A_HiPPO = -(A_gen + np.eye(N)).T
        elif a_tilt == "translate":
            A_HiPPO = A_gen - (self.edge_kappa / window) * np.outer(phi_1, phi_1)
        elif a_tilt == "translate_legacy":
            A_HiPPO = -A_gen
        elif a_tilt == "raw":
            A_HiPPO = A_gen
        else:
            raise ValueError(
                f"Unknown a_tilt={a_tilt!r}; expected 'legs', 'translate', "
                "'translate_legacy', or 'raw'."
            )
        if B_vec is not None:
            B = np.asarray(B_vec, dtype=np.float64)
        elif a_tilt == "translate":
            B = (self.edge_kappa / window) * phi_1
        else:
            B = phi_1

        self.A = A_HiPPO
        self.B = B
        self.measure_fn = self.measure()

        C = np.ones((1, N))
        D = np.zeros((1,))

        dA, dB, _, _, _ = signal.cont2discrete(
            (A_HiPPO, B, C, D), dt=dt, method=discretization
        )

        # Use A_infty if specified. (Our end-to-end A)
        if derive_A_from == "infty":
            A_infty = compute_A_infty(
                N,
                self.phi_func,
                self.sigma_func,
                self.sigma_inv_func,
                dt=self.dt,
                quad_order=self.quad_order,
                z_breakpoints=z_breakpoints,
            )
            # inv for well-conditioned A_infty (LegS, unchanged); pinv when singular
            # (e.g. FouT's null sin(0) mode).
            if np.linalg.cond(A_infty) < 1e12:
                A_infty_corrected = np.linalg.inv(A_infty.T) * np.exp(-dt)
            else:
                A_infty_corrected = np.linalg.pinv(A_infty.T) * np.exp(-dt)
            dA = A_infty_corrected
            print("\nLagOp HiPPO: Using A_infty for discretized A")

        # Use B_zoh if specified. (Our end-to-end B)
        if input_hold == "zoh" and a_tilt == "translate" and derive_A_from == "gen":
            # Edge-closed (A, B) already contain the closure's share of u(t); the
            # strip integral would drop it. Use the exact ZOH of the continuous pair.
            _, dB, _, _, _ = signal.cont2discrete(
                (A_HiPPO, B[:, None], C, D), dt=dt, method="zoh"
            )
            dB = dB[:, 0]
        elif input_hold == "zoh":
            B_zoh = compute_B_zoh(
                N,
                self.phi_func,
                dt=self.dt,
                quad_order=self.quad_order,
                sigma_func=self.sigma_func,
            )
            dB = B_zoh
            print("\nLagOp HiPPO: Using ZOH for discretized B:")

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
        if self.mode == "translate":
            # Truncate basis outside the active translate window.
            window_weight = self.sigma_deriv_func(0.0, -self.vals)
            eval_matrix[window_weight <= 0.0] = 0.0
        decay_factor = np.exp(-self.c * self.vals)  # (len(vals),)
        eval_matrix = eval_matrix * decay_factor[:, np.newaxis]  # Broadcast decay
        end_time = time.time()
        print(
            f"LagOp HiPPO: Total computation time for basis: {end_time - start_time:.2f} seconds"
        )

        return torch.Tensor(eval_matrix)

    def measure(self):
        return lambda x: self.sigma_deriv_func(0, -x)

    def _compute_measure(self):
        return self.measure()(self.vals)

    def forward(self, inputs, fast=True, verbose=False):
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
            if verbose:
                print(
                    f"LagOp HiPPO: Total computation time for fast forward pass: {time_end - time_start:.2f} seconds"
                )
            return result

        c = torch.zeros(u.shape[1:], device=inputs.device)
        cs = []
        for f in inputs:
            c = torch.matmul(self.dA, c.unsqueeze(-1)).squeeze(-1) + self.dB * f
            cs.append(c)
        time_end = time.time()
        if verbose:
            print(
                f"LagOp HiPPO: Total computation time for forward pass: {time_end - time_start:.2f} seconds"
            )
        return torch.stack(cs, dim=0)

    def reconstruct(self, c, evals=None):
        """
        c: (..., N,) HiPPO coefficients
        output: (..., L,)
        """
        if evals is not None:
            z_warped = self.sigma_func(0, -evals)
            eval_matrix = vectorized_basis(self.phi_func, np.arange(self.N), z_warped)
            decay_factor = np.exp(-self.c * evals)
            eval_matrix = eval_matrix * decay_factor[:, np.newaxis]
            eval_matrix = torch.Tensor(eval_matrix)
        else:
            eval_matrix = self.eval_matrix

        c = c.unsqueeze(-1)
        y = torch.matmul(eval_matrix.to(c.device), c)
        return y.squeeze(-1).flip(-1)


# Synthetic data generation
def whitesignal(period, dt, freq, rms=0.5, batch_shape=()):
    """
    Produces output signal of length period / dt, band-limited to frequency freq
    Output shape (*batch_shape, period/dt)
    Adapted from the nengo library
    """
    if freq is not None and freq < 1.0 / period:
        raise ValueError(
            f"Make ``{freq=} >= 1. / {period=}`` to produce a non-zero signal",
        )

    nyquist_cutoff = 0.5 / dt
    if freq > nyquist_cutoff:
        raise ValueError(
            f"{freq} must not exceed the Nyquist frequency for the given dt ({nyquist_cutoff:0.3f})"
        )

    n_coefficients = int(np.ceil(period / dt / 2.0))
    shape = batch_shape + (n_coefficients + 1,)
    sigma = rms * np.sqrt(0.5)
    coefficients = 1j * np.random.normal(0.0, sigma, size=shape)
    coefficients[..., -1] = 0.0
    coefficients += np.random.normal(0.0, sigma, size=shape)
    coefficients[..., 0] = 0.0

    set_to_zero = np.fft.rfftfreq(2 * n_coefficients, d=dt) > freq
    coefficients *= 1 - set_to_zero
    power_correction = np.sqrt(1.0 - np.sum(set_to_zero, dtype=float) / n_coefficients)
    if power_correction > 0.0:
        coefficients /= power_correction
    coefficients *= np.sqrt(2 * n_coefficients)
    signal = np.fft.irfft(coefficients, axis=-1)
    signal = signal - signal[..., :1]  # Start from 0
    return signal


def lorenzsignal(
    period, dt, batch_shape=(), initial_state=None, sigma=10.0, rho=28.0, beta=8.0 / 3.0
):
    """
    Generates a signal from the Lorenz attractor.
    Output shape (*batch_shape, period/dt, 3) for the x, y, z coordinates.
    """
    n_steps = int(np.ceil(period / dt))

    # If no initial state is provided, create a random one near the origin
    if initial_state is None:
        initial_state = np.random.randn(*batch_shape, 3) * 0.5 + np.array(
            [0.0, 1.0, 1.05]
        )
    else:
        # Ensure initial state can be broadcast to the batch shape
        initial_state = np.broadcast_to(initial_state, batch_shape + (3,))

    # Pre-allocate array for the trajectory
    trajectory = np.zeros(batch_shape + (n_steps, 3))
    trajectory[..., 0, :] = initial_state

    # Integrate using the Euler method
    for t in range(1, n_steps):
        # Get the current state
        xyz = trajectory[..., t - 1, :]
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

        # Lorenz attractor differential equations
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        # Update the state
        trajectory[..., t, 0] = x + dx * dt
        trajectory[..., t, 1] = y + dy * dt
        trajectory[..., t, 2] = z + dz * dt

    return trajectory


############################
# ... (keep all imports and helper funcs as-is)


class LagHiPPOSplit(nn.Module):
    """Lag Operator-based time invariant x_{t+1} = A x_t + B u_t"""

    def __init__(
        self,
        N,
        phi_func,
        phi_deriv_func,
        sigma_func,  # Now list for split [fast, slow]
        sigma_deriv_func,  # List
        sigma_inv_func,  # List
        sigma_inv_deriv_func,  # List
        dt=1.0,
        T=1.0,
        c=0.0,
        discretization="bilinear",
        derive_A_from="gen",
        mode="scale",
        window=1.0,
        input_hold="delta",
        split_sizes=None,  # New: e.g., [N//2, N//2]
    ):
        super().__init__()
        self.N = N
        self.dt = dt
        self.T = T
        self.c = c
        self.phi_func = phi_func
        self.phi_deriv_func = phi_deriv_func
        self.mode = mode
        self.window = window

        if split_sizes is None:
            split_sizes = [N]  # Single warp
            sigma_func = (
                [sigma_func] if not isinstance(sigma_func, list) else sigma_func
            )
            # ... (similar for others)

        assert sum(split_sizes) == N, "Split sizes must sum to N"
        self.split_sizes = split_sizes
        self.sigma_funcs = (
            sigma_func
            if isinstance(sigma_func, list)
            else [sigma_func] * len(split_sizes)
        )
        self.sigma_deriv_funcs = (
            sigma_deriv_func
            if isinstance(sigma_deriv_func, list)
            else [sigma_deriv_func] * len(split_sizes)
        )
        self.sigma_inv_funcs = (
            sigma_inv_func
            if isinstance(sigma_inv_func, list)
            else [sigma_inv_func] * len(split_sizes)
        )
        self.sigma_inv_deriv_funcs = (
            sigma_inv_deriv_func
            if isinstance(sigma_inv_deriv_func, list)
            else [sigma_inv_deriv_func] * len(split_sizes)
        )

        # Compute per-split A_gen/B, combine block-diag
        A_gen_blocks = []
        B_blocks = []
        for i, N_split in enumerate(split_sizes):
            A_gen_i = compute_A_gen(
                N_split,
                self.phi_func,
                self.phi_deriv_func,
                dt=self.dt,
                sigma_inv_deriv_func=self.sigma_inv_deriv_funcs[i],
            )
            B_i = compute_B_delta(N_split, self.phi_func)
            A_gen_blocks.append(A_gen_i)
            B_blocks.append(B_i)

        A_gen = np.block(
            [
                [
                    (
                        A_gen_blocks[i]
                        if i == j
                        else np.zeros((self.split_sizes[i], self.split_sizes[j]))
                    )
                    for j in range(len(split_sizes))
                ]
                for i in range(len(split_sizes))
            ]
        )
        B = np.concatenate(B_blocks)

        A_HiPPO = -(A_gen + np.eye(N)).T  # Apply tilt
        self.A = A_HiPPO
        self.B = B
        self.measure_fn = self.measure()  # Use first warp for demo; adapt if needed

        C = np.ones((1, N))
        D = np.zeros((1,))

        dA, dB, _, _, _ = signal.cont2discrete(
            (A_HiPPO, B, C, D), dt=dt, method=discretization
        )

        # ... (keep derive_A_from, input_hold as-is; for split, apply to blocks if needed)

        self.register_buffer("dA", torch.Tensor(dA))  # (N, N)
        self.register_buffer("dB", torch.Tensor(dB))  # (N,)

        # Evaluation matrix: Compute per-split and concatenate
        self.vals = np.arange(0.0, T, dt)
        self.eval_matrix = self._compute_eval_matrix()  # (len(vals), N)
        self.measure = self._compute_measure()  # Use first for demo

    def _compute_eval_matrix(self):
        eval_blocks = []
        for i in range(len(self.split_sizes)):
            z_warped = self.sigma_funcs[i](0, -self.vals)
            n_vals = np.arange(
                sum(self.split_sizes[:i]), sum(self.split_sizes[: i + 1])
            )
            eval_block = vectorized_basis(self.phi_func, n_vals, z_warped)
            decay_factor = np.exp(-self.c * self.vals)
            eval_block *= decay_factor[:, np.newaxis]
            eval_blocks.append(eval_block)
        return torch.Tensor(np.hstack(eval_blocks))

    # ... (keep forward/reconstruct as-is; they work with combined A/B/eval)


# # %%
# # Example usage for toy
# model_split = LagHiPPOSplit(
#     N=4,
#     phi_func=phi,
#     phi_deriv_func=phi_deriv,
#     sigma_func=[sigma_fast, sigma_slow],
#     sigma_deriv_func=[sigma_deriv_fast, sigma_deriv_slow],
#     sigma_inv_func=[sigma_inv_fast, sigma_inv_slow],
#     sigma_inv_deriv_func=[sigma_inv_deriv_fast, sigma_inv_deriv_slow],
#     dt=0.01,
#     T=10.0,
#     split_sizes=[2, 2],
# )

# # Standard model for comparison
# model_standard = LagHiPPO(
#     N=4,
#     phi_func=phi,
#     phi_deriv_func=phi_deriv,
#     sigma_func=sigma,  # Original
#     sigma_deriv_func=sigma_deriv,
#     sigma_inv_func=sigma_inv,
#     sigma_inv_deriv_func=sigma_inv_deriv,
#     dt=0.01,
#     T=10.0,
# )

# # Test on mixed signal (adapt your lorenz or whitesignal for mixed freq)
# vals = np.arange(0.0, 10.0, 0.01)
# u = np.sin(0.5 * 2 * np.pi * vals) + np.sin(10 * 2 * np.pi * vals)  # Low + high
# u_torch = torch.tensor(u, dtype=torch.float)

# c_split = model_split.forward(u_torch)
# recon_split = model_split.reconstruct(c_split[-1]).numpy()

# c_standard = model_standard.forward(u_torch)
# recon_standard = model_standard.reconstruct(c_standard[-1]).numpy()

# mse_split = np.mean((u - recon_split) ** 2)
# mse_standard = np.mean((u - recon_standard) ** 2)

# print(f"MSE Split: {mse_split}, MSE Standard: {mse_standard}")
