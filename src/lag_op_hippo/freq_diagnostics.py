"""Diagnostics utilities for frequency-preserving basis experiments."""

from dataclasses import dataclass
import time
import numpy as np
from scipy.interpolate import interp1d

from lag_op_hippo.lag_hippo import get_gauss_quadrature, vectorized_basis


@dataclass
class ProjectionReconstructionResult:
    coefficients: np.ndarray
    reconstruction: np.ndarray
    mse: float
    runtime_s: float


def build_interpolated_signal(s_vals, u_vals, kind="linear"):
    """Create u(s) callable from discrete samples with zero fill outside bounds."""
    return interp1d(
        s_vals,
        u_vals,
        kind=kind,
        bounds_error=False,
        fill_value=0.0,
    )


def compute_inner_product(phi_func, u_func, sigma_inv_func, N, t=0.0, quad_order=None):
    """Compute c_n = integral phi_n(z) * u(sigma_inv_t(z)) dz using Gauss quadrature."""
    if quad_order is None:
        quad_order = max(2 * N, 4)
    z_nodes, weights = get_gauss_quadrature(quad_order, 0.0, 1.0)
    s_vals = sigma_inv_func(t, z_nodes)
    phi_vals = vectorized_basis(phi_func, np.arange(N), z_nodes)
    u_vals = u_func(s_vals)
    return np.einsum("i,ij,i->j", weights, phi_vals, u_vals)


def reconstruct_signal(s_vals, c, phi_func, sigma_func, t=0.0):
    """Reconstruct u(s) from coefficients c and basis psi_{t,n}(s)."""
    z_vals = sigma_func(t, s_vals)
    phi_vals = vectorized_basis(phi_func, np.arange(len(c)), z_vals)
    return phi_vals @ c


def run_projection_reconstruction(
    u_func,
    s_vals,
    phi_func,
    sigma_func,
    sigma_inv_func,
    N,
    t=0.0,
    quad_order=None,
    verbose=True,
):
    """Project a signal then reconstruct it over s_vals."""
    start_time = time.time()
    coefficients = compute_inner_product(
        phi_func,
        u_func,
        sigma_inv_func,
        N,
        t=t,
        quad_order=quad_order,
    )
    reconstruction = reconstruct_signal(s_vals, coefficients, phi_func, sigma_func, t=t)
    target = u_func(s_vals)
    mse = float(np.mean((reconstruction - target) ** 2))
    runtime_s = time.time() - start_time
    if verbose:
        print(f"Projection-reconstruction completed in {runtime_s:.4f}s")
        print(f"MSE: {mse:.6e}")
    return ProjectionReconstructionResult(
        coefficients=coefficients,
        reconstruction=reconstruction,
        mse=mse,
        runtime_s=runtime_s,
    )


def orthonormality_matrix_gauss(N, phi_func, Z=(0.0, 1.0), quad_order=1000):
    """Return inner-product matrix <phi_i, phi_j> over interval Z."""
    z_nodes, weights = get_gauss_quadrature(quad_order, Z[0], Z[1])
    phi_vals = vectorized_basis(phi_func, np.arange(N), z_nodes)
    return np.einsum("i,ik,il->kl", weights, phi_vals, phi_vals)


def orthonormality_summary(inner_mat, tolerance=1e-5):
    """Summarize orthonormality quality for an inner-product matrix."""
    if inner_mat.size == 0:
        return {
            "is_ok": True,
            "diag_reference": 0.0,
            "max_diag_deviation": 0.0,
            "max_offdiag": 0.0,
            "tolerance": tolerance,
        }

    diag = np.diag(inner_mat)
    diag_reference = float(diag[0])
    max_diag_deviation = float(np.max(np.abs(diag - diag_reference)))
    offdiag = inner_mat - np.diag(diag)
    max_offdiag = float(np.max(np.abs(offdiag)))
    is_ok = max_diag_deviation <= tolerance and max_offdiag <= tolerance
    return {
        "is_ok": bool(is_ok),
        "diag_reference": diag_reference,
        "max_diag_deviation": max_diag_deviation,
        "max_offdiag": max_offdiag,
        "tolerance": tolerance,
    }


__all__ = [
    "ProjectionReconstructionResult",
    "build_interpolated_signal",
    "compute_inner_product",
    "reconstruct_signal",
    "run_projection_reconstruction",
    "orthonormality_matrix_gauss",
    "orthonormality_summary",
]
