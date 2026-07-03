"""Frequency-preserving HiPPO basis and model builders.

This module is intentionally side-effect free so it can be imported by
interactive notebooks, scripts, and tests without running experiments.
"""

from dataclasses import dataclass
import numpy as np

from lag_op_hippo.lag_hippo import LagHiPPO


@dataclass(frozen=True)
class FrequencyPreservingConfig:
    """Configuration for the piecewise frequency-preserving warp."""

    alpha: float = 0.2
    window_length: float = 1.0
    eps: float = 1e-8

    def __post_init__(self):
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must satisfy 0 < alpha < 1")
        if self.window_length <= 0.0:
            raise ValueError("window_length must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


class FrequencyPreservingBasis:
    """Callable collection for sigma/phi and derivatives."""

    def __init__(self, config: FrequencyPreservingConfig | None = None):
        self.config = config or FrequencyPreservingConfig()

    def _window_index(self, tau):
        tau = np.asarray(tau)
        tau = np.clip(tau, 0.0, np.inf)
        return (np.floor(tau / self.config.window_length) + 1).astype(int)

    def f_tau(self, tau):
        """Map tau >= 0 to z in [0, 1]."""
        n = self._window_index(tau)
        alpha_n = self.config.alpha**n
        h_n = (1.0 - self.config.alpha) * (self.config.alpha ** (n - 1))
        frac = n - np.asarray(tau) / self.config.window_length
        z = alpha_n + h_n * frac
        return np.clip(z, 0.0, 1.0)

    def sigma(self, t, s):
        return self.f_tau(np.asarray(t) - np.asarray(s))

    def sigma_deriv(self, t, s):
        tau = np.asarray(t) - np.asarray(s)
        n = self._window_index(tau)
        h_n = (1.0 - self.config.alpha) * (self.config.alpha ** (n - 1))
        return h_n / self.config.window_length

    def sigma_inv(self, t, z):
        z = np.asarray(z)
        z = np.clip(z, self.config.eps, 1.0)
        p = np.log(z) / np.log(self.config.alpha)
        n = (np.floor(p) + 1).astype(int)
        alpha_n = self.config.alpha**n
        h_n = (1.0 - self.config.alpha) * (self.config.alpha ** (n - 1))
        tau = self.config.window_length * (n - (z - alpha_n) / h_n)
        return np.asarray(t) - tau

    def sigma_inv_deriv(self, t, z):
        del t
        z = np.asarray(z)
        z = np.clip(z, self.config.eps, 1.0)
        p = np.log(z) / np.log(self.config.alpha)
        n = (np.floor(p) + 1).astype(int)
        h_n = (1.0 - self.config.alpha) * (self.config.alpha ** (n - 1))
        return self.config.window_length / h_n

    def psi(self, n, s):
        k = n + 1
        m = k // 2
        tau = np.asarray(s)
        n_win = self._window_index(tau)
        h_n = (1.0 - self.config.alpha) * (self.config.alpha ** (n_win - 1))
        amp = np.sqrt(1.0 / h_n)
        arg = 2.0 * np.pi * m * np.asarray(s) / self.config.window_length
        if k % 2 == 1:
            return amp * np.cos(arg)
        return amp * np.sin(arg)

    def phi(self, n, z):
        s = self.sigma_inv(0.0, z)
        return self.psi(n, s)

    def psi_deriv(self, n, s):
        k = n + 1
        m = k // 2
        tau = np.asarray(s)
        n_win = self._window_index(tau)
        h_n = (1.0 - self.config.alpha) * (self.config.alpha ** (n_win - 1))
        amp = np.sqrt(1.0 / h_n)
        arg = 2.0 * np.pi * m * np.asarray(s) / self.config.window_length
        darg_ds = 2.0 * np.pi * m / self.config.window_length
        if k % 2 == 1:
            return -amp * np.sin(arg) * darg_ds
        return amp * np.cos(arg) * darg_ds

    def phi_deriv(self, n, z):
        s = self.sigma_inv(0.0, z)
        ds_dz = self.sigma_inv_deriv(0.0, z)
        return self.psi_deriv(n, s) * ds_dz


_DEFAULT_BASIS = FrequencyPreservingBasis()


def f_tau(tau):
    """Backward-compatible alias using default config."""
    return _DEFAULT_BASIS.f_tau(tau)


def sigma_fourier(t, s):
    """Backward-compatible alias using default config."""
    return _DEFAULT_BASIS.sigma(t, s)


def sigma_deriv_fourier(t, s):
    """Backward-compatible alias using default config."""
    return _DEFAULT_BASIS.sigma_deriv(t, s)


def sigma_inv_fourier(t, z):
    """Backward-compatible alias using default config."""
    return _DEFAULT_BASIS.sigma_inv(t, z)


def sigma_inv_deriv_fourier(t, z):
    """Backward-compatible alias using default config."""
    return _DEFAULT_BASIS.sigma_inv_deriv(t, z)


def psi_fourier(n, s):
    """Backward-compatible alias using default config."""
    return _DEFAULT_BASIS.psi(n, s)


def phi_fourier(n, z):
    """Backward-compatible alias using default config."""
    return _DEFAULT_BASIS.phi(n, z)


def psi_deriv_fourier(n, s):
    """Backward-compatible alias using default config."""
    return _DEFAULT_BASIS.psi_deriv(n, s)


def phi_deriv_fourier(n, z):
    """Backward-compatible alias using default config."""
    return _DEFAULT_BASIS.phi_deriv(n, z)


def make_frequency_preserving_basis(
    alpha: float = 0.2,
    window_length: float = 1.0,
    eps: float = 1e-8,
):
    """Create a parameterized frequency-preserving basis object."""
    config = FrequencyPreservingConfig(
        alpha=alpha,
        window_length=window_length,
        eps=eps,
    )
    return FrequencyPreservingBasis(config=config)


def make_frequency_preserving_model(
    N: int,
    dt: float,
    T: float,
    alpha: float = 0.2,
    window_length: float = 1.0,
    eps: float = 1e-8,
    derive_A_from: str = "infty",
    input_hold: str = "zoh",
    quad_order: int | None = None,
):
    """Build a LagHiPPO model configured with frequency-preserving basis."""
    basis = make_frequency_preserving_basis(
        alpha=alpha,
        window_length=window_length,
        eps=eps,
    )
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
    )


__all__ = [
    "FrequencyPreservingConfig",
    "FrequencyPreservingBasis",
    "f_tau",
    "sigma_fourier",
    "sigma_deriv_fourier",
    "sigma_inv_fourier",
    "sigma_inv_deriv_fourier",
    "psi_fourier",
    "phi_fourier",
    "psi_deriv_fourier",
    "phi_deriv_fourier",
    "make_frequency_preserving_basis",
    "make_frequency_preserving_model",
]
