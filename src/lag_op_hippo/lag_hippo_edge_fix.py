"""Opt-in fixes for LagOp translate (LegT/FouT) and FouP families (added 2026-11-19).

Nothing in this module changes any existing default: the old builders in
``lag_hippo``, ``lag_hippo_translated`` and ``lag_hippo_freq`` are untouched, so
earlier experiments reproduce byte-for-byte.  Use these functions explicitly.

Fixes provided
--------------
1. Warp-correct ``A_gen`` (``compute_A_gen_for_basis``) instead of
   ``lag_hippo.compute_A_gen``, which hard-codes the LegS ``sigma_inv_deriv``.
2. Rank-one edge term for truncated (translate) windows.  For a window of
   length ``L`` with present at ``z=1`` and oldest edge at ``z=0``:

       dc/dt = A_gen c + (1/L)[phi(1) u(t) - phi(0) u(t-L)]        (exact)

   (integration by parts, using phi(1)phi(1)^T - phi(0)phi(0)^T = M + M^T).
   Closing ``u(t-L)`` with the reconstruction at the edge gives

       LegT : u(t-L) ~ phi(0)^T c            -> A = A_gen - (1/L) phi(1)phi(1)^T,  B = phi(1)/L
       FouT : u(t-L) ~ 2 phi(0)^T c - u(t)   -> A = A_gen - (2/L) phi(1)phi(1)^T,  B = 2 phi(1)/L
             (Fourier partial sums converge to the mid-point of the jump; phi(0)=phi(1)).
   i.e. A = A_gen - (kappa/L) phi(1)phi(1)^T, B = (kappa/L) phi(1), kappa_LegT=1, kappa_FouT=2.
   This reproduces Gu's transition('legt'/'fourier') to ~1e-12.
3. FouP basis with correct normalisation (``FrequencyPreservingBasisFixed``):
   the repo ``FrequencyPreservingBasis.phi`` passes s = sigma_inv(0,z) <= 0 into
   ``psi``, whose window index clips to window 1, so the amplitude is the
   constant 1/sqrt(h_1) and Gram = diag(1.25, 0.625, ...).  Here phi_k(z) =
   g_k(frac(tau(z)/L)) with the Gu-ordered orthonormal Fourier vector g
   (1, 0, sqrt2 cos, sqrt2 sin, ...), which is orthonormal w.r.t. dz on [0,1]
   because sum_n h_n = 1.
4. FouP edge closure: the FouP measure jumps by a factor alpha at every window
   edge tau = nL, giving
       dc/dt = (1/L)[D c + g0 (h_1 u(t) - (1-alpha) sum_n h_n u(t-nL))].
   With the mid-point closure on the alpha-weighted fold this is
       A = (D - kappa g0 g0^T)/L,  B = kappa g0/L,  kappa = 2(1-alpha)/(1+alpha)
   (kappa -> 2, i.e. FouT, as alpha -> 0).  The block cascade (one FouT block
   per window, lower-bidiagonal / lower-triangular coupling) is in
   ``window_checks.basis_utils.block_routes``.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import signal

from lag_op_hippo.freq_A_discrepancy_checks import compute_A_gen_for_basis
from lag_op_hippo.lag_hippo_freq import FrequencyPreservingBasis, FrequencyPreservingConfig
from lag_op_hippo.lag_hippo_translated import (
    _make_lag_model, fout_input_vector, legt_input_vector, make_fout_basis, make_legt_basis)

EDGE_KAPPA = {"legt": 1.0, "fourier": 2.0}


def foup_edge_kappa(alpha: float) -> float:
    return 2.0 * (1.0 - alpha) / (1.0 + alpha)


def translate_edge_A_B(basis, N: int, kappa: float, window_length: float = 1.0, quad_order: int | None = None):
    """A = A_gen(warp-correct) - (kappa/L) phi(1)phi(1)^T, B = (kappa/L) phi(1)."""
    qo = quad_order or max(200 * N, 64)
    A_gen = compute_A_gen_for_basis(N, basis.phi, basis.phi_deriv, basis.sigma_inv_deriv, quad_order=qo)
    p1 = np.array([float(np.asarray(basis.phi(k, np.array([1.0])))[0]) for k in range(N)])
    A = A_gen - (kappa / window_length) * np.outer(p1, p1)
    B = (kappa / window_length) * p1
    return A, B, A_gen, p1


def make_translate_model_edge_fixed(family: str, N: int, dt: float, T: float, window_length: float = 1.0,
                                    quad_order: int | None = None, discretization: str = "bilinear"):
    """LagHiPPO for 'legt' or 'fourier' with the edge-fixed (A, B); same eval matrix as the repo builder."""
    if family == "legt":
        basis, Bv = make_legt_basis(window_length=window_length), legt_input_vector(N)
    elif family == "fourier":
        basis, Bv = make_fout_basis(window_length=window_length), fout_input_vector(N)
    else:
        raise ValueError(family)
    model = _make_lag_model(basis, N=N, dt=dt, T=T, B_vec=Bv)
    A, B, A_gen, p1 = translate_edge_A_B(basis, N, EDGE_KAPPA[family], window_length, quad_order)
    dA, dB, *_ = signal.cont2discrete((A, B[:, None], np.ones((1, N)), np.zeros((1,))), dt=dt, method=discretization)
    model.A, model.B = A, B
    model.dA = torch.tensor(dA, dtype=torch.float)
    model.dB = torch.tensor(dB[:, 0], dtype=torch.float)
    model.a_tilt = "translate_edge"
    model.A_gen_warp_correct = A_gen
    model.dA64, model.dB64 = dA, dB[:, 0]
    return model


def gu_fourier_vector(x, N: int):
    """Gu-ordered orthonormal Fourier functions on x in [0,1): (len(x), N)."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.zeros((x.size, N))
    out[:, 0] = 1.0
    for m in range(1, N // 2):
        out[:, 2 * m] = np.sqrt(2) * np.cos(2 * np.pi * m * x)
        out[:, 2 * m + 1] = np.sqrt(2) * np.sin(2 * np.pi * m * x)
    return out


def gu_fourier_vector_deriv(x, N: int):
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.zeros((x.size, N))
    for m in range(1, N // 2):
        w = 2 * np.pi * m
        out[:, 2 * m] = -np.sqrt(2) * w * np.sin(w * x)
        out[:, 2 * m + 1] = np.sqrt(2) * w * np.cos(w * x)
    return out


class FrequencyPreservingBasisFixed(FrequencyPreservingBasis):
    """FouP warp of the repo, with an orthonormal (Gu-ordered) unchirped Fourier basis.

    Warp functions (sigma, sigma_inv, ...) are inherited unchanged; only phi/phi_deriv differ.
    Note the Gu ordering keeps the null sin(0) slot at index 1 (as HiPPO-FouT)."""

    def __init__(self, N: int, config: FrequencyPreservingConfig | None = None):
        super().__init__(config)
        self.N = N

    def _x(self, z):
        tau = -np.asarray(self.sigma_inv(0.0, z), dtype=np.float64)
        return np.mod(tau / self.config.window_length, 1.0)

    def phi(self, n, z):
        scalar = np.ndim(z) == 0
        v = gu_fourier_vector(self._x(z), self.N)[:, n]
        return v[0] if scalar else v

    def phi_deriv(self, n, z):
        z = np.asarray(z, dtype=np.float64)
        dtau_dz = -np.asarray(self.sigma_inv_deriv(0.0, z), dtype=np.float64)
        return gu_fourier_vector_deriv(self._x(z), self.N)[:, n] / self.config.window_length * dtau_dz


def make_frequency_preserving_basis_fixed(N: int, alpha: float = 0.2, window_length: float = 1.0, eps: float = 1e-8):
    return FrequencyPreservingBasisFixed(N, FrequencyPreservingConfig(alpha=alpha, window_length=window_length, eps=eps))


__all__ = ["EDGE_KAPPA", "foup_edge_kappa", "translate_edge_A_B", "make_translate_model_edge_fixed",
           "gu_fourier_vector", "gu_fourier_vector_deriv", "FrequencyPreservingBasisFixed",
           "make_frequency_preserving_basis_fixed"]
