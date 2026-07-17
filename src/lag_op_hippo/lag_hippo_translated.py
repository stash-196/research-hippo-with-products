"""Truncated-translate LagOp bases replicating HiPPO-LegT and HiPPO-FouT.

Uses a single-window linear time warp (no cross-window decay) together with
Legendre or Fourier bases in the normalized coordinate z.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import eval_legendre

from lag_op_hippo.lag_hippo import LagHiPPO, phi_deriv as _legendre_phi_deriv


@dataclass(frozen=True)
class TruncatedTranslateConfig:
    """Configuration for the first-interval truncated translate warp."""

    window_length: float = 1.0
    eps: float = 1e-8

    def __post_init__(self):
        if self.window_length <= 0.0:
            raise ValueError("window_length must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


class TruncatedTranslateBasis:
    """Truncated linear warp on the first window only.

    Lookback ``tau = t - s`` is clipped to ``[0, L]`` and mapped linearly to
    ``z = 1 - tau / L`` so the present (``tau = 0``) maps to ``z = 1`` and the
    oldest retained history (``tau = L``) maps to ``z = 0``.  History older than
    ``L`` is truncated (``z = 0``, measure zero).

    This is the alpha -> 0, single-window limit of the frequency-preserving
    warp without exponential decay across windows.
    """

    def __init__(self, config: TruncatedTranslateConfig | None = None):
        self.config = config or TruncatedTranslateConfig()

    @property
    def L(self) -> float:
        return self.config.window_length

    def _tau(self, t, s):
        return np.clip(np.asarray(t, dtype=np.float64) - np.asarray(s, dtype=np.float64), 0.0, self.L)

    def sigma(self, t, s):
        tau = self._tau(t, s)
        return 1.0 - tau / self.L

    def sigma_deriv(self, t, s):
        tau = np.asarray(t, dtype=np.float64) - np.asarray(s, dtype=np.float64)
        inside = (tau >= 0.0) & (tau <= self.L)
        out = np.zeros_like(tau, dtype=np.float64)
        out[inside] = 1.0 / self.L
        return out

    def sigma_inv(self, t, z):
        z = np.asarray(z, dtype=np.float64)
        z = np.clip(z, 0.0, 1.0)
        tau = self.L * (1.0 - z)
        return np.asarray(t, dtype=np.float64) - tau

    def sigma_inv_deriv(self, t, z):
        del t
        z = np.asarray(z, dtype=np.float64)
        z = np.clip(z, self.config.eps, 1.0)
        return np.full_like(z, self.L, dtype=np.float64)


class LegTBasis(TruncatedTranslateBasis):
    """Legendre basis on z, faithful to Gu HiPPO-LegT on lookback.

    Warp is ``z = 1 - tau/L`` (present at ``z=1``). Gu evaluates
    ``sqrt(2n+1) (-1)^n P_n(2 tau - 1)`` on lookback ``tau``.  Substituting
    ``tau = L(1-z)`` (unit window ``L=1``) and Legendre parity yields exactly

        phi_n(z) = sqrt(2n+1) P_n(2z - 1)

    i.e. **no** extra ``(-1)^n`` on top of the flipped warp (that factor is
    already accounted for by ``z = 1 - tau``).
    """

    def phi(self, n, z):
        z = np.asarray(z, dtype=np.float64)
        y = 2.0 * z - 1.0
        return np.sqrt(2 * n + 1) * eval_legendre(n, y)

    def phi_deriv(self, n, z):
        return _legendre_phi_deriv(n, z)


class FouTBasis(TruncatedTranslateBasis):
    """Fourier (cos/sin) basis on z, faithful to Gu HiPPO-FouT on lookback.

    Gu evaluates cos/sin in **lookback** ``tau`` (present at 0).  With the
    LagOp warp ``z = 1 - tau/L`` we therefore use the lookback coordinate
    ``u = 1 - z`` (= ``tau/L``) inside the trig functions so that

        phi(z) = Gu_basis(tau = L(1-z))

    matches the original ``basis(..., method='fourier')`` eval matrix.
    """

    @staticmethod
    def _check_n_even(N: int) -> None:
        if N % 2 != 0:
            raise ValueError("FouT requires even N (cos/sin pairs)")

    def phi(self, n, z):
        z = np.asarray(z, dtype=np.float64)
        # Lookback-aligned coordinate (Gu): u = tau/L = 1 - z
        u = 1.0 - z
        if n == 0:
            return np.ones_like(z, dtype=np.float64)
        m = n // 2
        arg = 2.0 * np.pi * m * u
        if n % 2 == 0:
            return np.sqrt(2.0) * np.cos(arg)
        return np.sqrt(2.0) * np.sin(arg)

    def phi_deriv(self, n, z):
        # d/dz = d/du * du/dz with u = 1 - z, du/dz = -1
        z = np.asarray(z, dtype=np.float64)
        u = 1.0 - z
        if n == 0:
            return np.zeros_like(z, dtype=np.float64)
        m = n // 2
        arg = 2.0 * np.pi * m * u
        darg_du = 2.0 * np.pi * m
        if n % 2 == 0:
            # d/dz [sqrt2 cos(arg(u))] = -sqrt2 sin(arg) * darg_du * (-1)
            return np.sqrt(2.0) * np.sin(arg) * darg_du
        # d/dz [sqrt2 sin(arg(u))] = sqrt2 cos(arg) * darg_du * (-1)
        return -np.sqrt(2.0) * np.cos(arg) * darg_du


def make_truncated_translate_basis(window_length: float = 1.0, eps: float = 1e-8):
    config = TruncatedTranslateConfig(window_length=window_length, eps=eps)
    return TruncatedTranslateBasis(config=config)


def make_legt_basis(window_length: float = 1.0, eps: float = 1e-8) -> LegTBasis:
    config = TruncatedTranslateConfig(window_length=window_length, eps=eps)
    return LegTBasis(config=config)


def legt_input_vector(N: int) -> np.ndarray:
    """Input vector B matching HiPPO-LegT ``transition('legt')``."""
    return np.sqrt(2 * np.arange(N, dtype=np.float64) + 1.0)


def fout_input_vector(N: int) -> np.ndarray:
    """Input vector B matching HiPPO-FouT ``transition('fourier')``."""
    FouTBasis._check_n_even(N)
    B = np.zeros(N, dtype=np.float64)
    B[0::2] = 2.0
    B[0] = np.sqrt(2.0)
    B *= np.sqrt(2.0)
    return B


def make_fout_basis(window_length: float = 1.0, eps: float = 1e-8) -> FouTBasis:
    config = TruncatedTranslateConfig(window_length=window_length, eps=eps)
    return FouTBasis(config=config)


def _make_lag_model(
    basis: LegTBasis | FouTBasis,
    N: int,
    dt: float,
    T: float,
    B_vec: np.ndarray,
    derive_A_from: str = "gen",
    input_hold: str = "delta",
    quad_order: int | None = None,
) -> LagHiPPO:
    return LagHiPPO(
        N=N,
        phi_func=basis.phi,
        phi_deriv_func=basis.phi_deriv,
        sigma_func=basis.sigma,
        sigma_deriv_func=basis.sigma_deriv,
        sigma_inv_func=basis.sigma_inv,
        sigma_inv_deriv_func=basis.sigma_inv_deriv,
        dt=dt,
        T=T,
        derive_A_from=derive_A_from,
        input_hold=input_hold,
        quad_order=quad_order,
        mode="translate",
        window=basis.L,
        a_tilt="translate",
        B_vec=B_vec,
    )


def make_legt_model(
    N: int,
    dt: float,
    T: float,
    window_length: float = 1.0,
    eps: float = 1e-8,
    derive_A_from: str = "gen",
    input_hold: str = "delta",
    quad_order: int | None = None,
) -> LagHiPPO:
    """Build a LagHiPPO model configured to replicate HiPPO-LegT."""
    basis = make_legt_basis(window_length=window_length, eps=eps)
    return _make_lag_model(
        basis,
        N=N,
        dt=dt,
        T=T,
        B_vec=legt_input_vector(N),
        derive_A_from=derive_A_from,
        input_hold=input_hold,
        quad_order=quad_order,
    )


def make_fout_model(
    N: int,
    dt: float,
    T: float,
    window_length: float = 1.0,
    eps: float = 1e-8,
    derive_A_from: str = "gen",
    input_hold: str = "delta",
    quad_order: int | None = None,
) -> LagHiPPO:
    """Build a LagHiPPO model configured to replicate HiPPO-FouT."""
    FouTBasis._check_n_even(N)
    basis = make_fout_basis(window_length=window_length, eps=eps)
    return _make_lag_model(
        basis,
        N=N,
        dt=dt,
        T=T,
        B_vec=fout_input_vector(N),
        derive_A_from=derive_A_from,
        input_hold=input_hold,
        quad_order=quad_order,
    )


def compare_continuous_A_to_hippo(
    lag_model: LagHiPPO,
    hippo_A: np.ndarray,
) -> float:
    """Relative Frobenius error between LagOp continuous A and HiPPO reference A."""
    return float(
        np.linalg.norm(lag_model.A - hippo_A, ord="fro")
        / np.linalg.norm(hippo_A, ord="fro")
    )


def compare_discrete_A_to_hippo(
    lag_model: LagHiPPO,
    hippo_dA: np.ndarray,
) -> float:
    """Relative Frobenius error between LagOp dA and HiPPO reference dA."""
    return float(
        np.linalg.norm(lag_model.dA.numpy() - hippo_dA, ord="fro")
        / np.linalg.norm(hippo_dA, ord="fro")
    )


__all__ = [
    "TruncatedTranslateConfig",
    "TruncatedTranslateBasis",
    "LegTBasis",
    "FouTBasis",
    "make_truncated_translate_basis",
    "make_legt_basis",
    "make_fout_basis",
    "make_legt_model",
    "make_fout_model",
    "legt_input_vector",
    "fout_input_vector",
    "compare_continuous_A_to_hippo",
    "compare_discrete_A_to_hippo",
]