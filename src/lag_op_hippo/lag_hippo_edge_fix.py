"""Thin compatibility wrapper (since the 2026-11-19 library fix).

The fixes that this module introduced opt-in on 2026-11-19 now live in the
library itself (branch exp/legt-fout-fix-window-checks; pre-fix state is tag
``pre-libfix-2026-11-19``):

* ``lag_hippo.compute_A_gen`` takes the model's own ``sigma_inv_deriv`` (LagHiPPO
  passes it), instead of the hard-coded LegS warp.
* ``LagHiPPO(a_tilt='translate', edge_kappa=kappa)`` is the rank-one edge closure
  A = A_gen - (kappa/L) phi(1)phi(1)^T, B = (kappa/L) phi(1); ``make_legt_model``
  (kappa=1) / ``make_fout_model`` (kappa=2) use it by default and reproduce Gu's
  transition('legt'/'fourier') to ~1e-12.
* ``FrequencyPreservingBasis`` is orthonormal in z (amplitude 1 / sqrt2), and
  ``make_frequency_preserving_model`` defaults to the FouP edge closure with
  kappa = 2(1-alpha)/(1+alpha).

Background (unchanged):
   Exact (translate window, h = measure density = 1/L, G_{h'} = Gram under h'):
   dc/dt = -(A_gen^T + G_{h'}) c + h(1) phi(1) u(t) - h(0) phi(0) u(t-L)
         = A_gen c + h(1) phi(1)(u(t) - phi(1)^T c) - h(0) phi(0)(u(t-L) - phi(0)^T c).
   (The short form dc/dt = A_gen c + (1/L)[phi(1)u(t) - phi(0)u(t-L)] is exact only for
   FouT, where A_gen is skew and G_{h'} = 0; for LegT it is wrong, FD error ~8.)
   LegT : u(t-L) ~ phi(0)^T c            -> kappa = 1
   FouT : u(t-L) ~ 2 phi(0)^T c - u(t)   -> kappa = 2  (Fourier mid-point; phi(0)=phi(1))
   giving A = A_gen - (kappa/L) phi(1)phi(1)^T, B = (kappa/L) phi(1) in both cases.
   FouP : measure jumps by alpha per window edge -> kappa = 2(1-alpha)/(1+alpha).

What stays only here: ``FrequencyPreservingBasisFixed``, a FouP basis in Gu's
FouT ordering (1, 0, sqrt2 cos, sqrt2 sin, ...) with the null sin(0) slot, used by
``window_checks`` for block-by-block comparison with Gu FouT. The library keeps
the repo ordering (const, sin1, cos1, ...), which has no null mode.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from lag_op_hippo.lag_hippo import compute_A_gen, compute_B_delta
from lag_op_hippo.lag_hippo_freq import (FrequencyPreservingBasis, FrequencyPreservingConfig,
                                         foup_edge_kappa)
from lag_op_hippo.lag_hippo_translated import make_fout_model, make_legt_model

EDGE_KAPPA = {"legt": 1.0, "fourier": 2.0}


def translate_edge_A_B(basis, N: int, kappa: float, window_length: float = 1.0, quad_order: int | None = None):
    """A = A_gen(warp-correct) - (kappa/L) phi(1)phi(1)^T, B = (kappa/L) phi(1)."""
    qo = quad_order or max(200 * N, 64)
    A_gen = compute_A_gen(N, basis.phi, basis.phi_deriv, quad_order=qo, sigma_inv_deriv_func=basis.sigma_inv_deriv)
    p1 = compute_B_delta(N, basis.phi)
    A = A_gen - (kappa / window_length) * np.outer(p1, p1)
    B = (kappa / window_length) * p1
    return A, B, A_gen, p1


def make_translate_model_edge_fixed(family: str, N: int, dt: float, T: float, window_length: float = 1.0,
                                    quad_order: int | None = None, discretization: str = "bilinear"):
    """Now just the library builder (make_legt_model / make_fout_model), plus float64 copies."""
    if family == "legt":
        model = make_legt_model(N=N, dt=dt, T=T, window_length=window_length, quad_order=quad_order)
    elif family == "fourier":
        model = make_fout_model(N=N, dt=dt, T=T, window_length=window_length, quad_order=quad_order)
    else:
        raise ValueError(family)
    if discretization != "bilinear":
        raise ValueError("only bilinear (the library default) is supported")
    A, B = np.asarray(model.A, float), np.asarray(model.B, float)
    dA, dB, *_ = signal.cont2discrete((A, B[:, None], np.ones((1, N)), np.zeros((1,))), dt=dt, method="bilinear")
    model.A_gen_warp_correct = model.A_gen
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
