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


@dataclass
class WarpMeasureDensityResult:
    s_vals: np.ndarray
    measure_s: np.ndarray
    z_vals: np.ndarray
    measure_z: np.ndarray
    max_pushforward_deviation_from_1: float
    mean_pushforward_deviation_from_1: float
    z_min: float
    z_max: float
    is_uniform: bool


def compute_warp_measure_densities(
    sigma_fn,
    sigma_deriv_fn,
    sigma_inv_fn,
    sigma_inv_deriv_fn,
    t_ref,
    dt,
    z_eps=1e-12,
    uniform_tol=1e-8,
    verbose=False,
):
    """Compute measure densities on s and the pushed-forward density on z.

    LagOp projections use the time-domain measure
    ``omega_t(s) = sigma_t'(s)`` (the Jacobian ``dz/ds``).

    Change of variables ``z = sigma_t(s)`` gives
    ``∫ u(s) omega_t(s) ds = ∫ u(sigma_t^{-1}(z)) dz``,
    so the pushed-forward density on z must be uniform:
    ``omega_z(z) = sigma_t'(s(z)) * |ds/dz| = 1``.

    We evaluate this on the same s-grid used for the left panel and map
    forward to z = sigma_t(s) to avoid inverse-warp pairing errors.
    """
    s_vals = np.arange(0.0, t_ref, dt, dtype=np.float64)
    measure_s = np.asarray(sigma_deriv_fn(t_ref, s_vals), dtype=np.float64)

    z_vals = np.asarray(sigma_fn(t_ref, s_vals), dtype=np.float64)
    z_safe = np.maximum(z_vals, z_eps)

    # dz/ds at the paired s point, and |ds/dz| at the paired z point.
    dz_ds = np.maximum(np.asarray(sigma_deriv_fn(t_ref, s_vals), dtype=np.float64), 0.0)
    ds_dz = np.abs(np.asarray(sigma_inv_deriv_fn(t_ref, z_safe), dtype=np.float64))

    measure_z = dz_ds * ds_dz

    # Independent check via inverse warp (should agree up to fp error).
    s_from_z = np.asarray(sigma_inv_fn(t_ref, z_safe), dtype=np.float64)
    measure_z_via_inv = (
        np.asarray(sigma_deriv_fn(t_ref, s_from_z), dtype=np.float64) * ds_dz
    )
    inv_agreement = float(np.max(np.abs(measure_z - measure_z_via_inv)))

    # For truncated warps many s values map to z=0 with zero Jacobian; only
    # assess uniformity on the active window where dz/ds > 0.
    active = dz_ds > 1e-12
    if not np.any(active):
        raise ValueError("No active measure support found for warp diagnostics.")
    measure_z_active = measure_z[active]
    z_active = z_vals[active]

    max_dev = float(np.max(np.abs(measure_z_active - 1.0)))
    mean_dev = float(np.mean(np.abs(measure_z_active - 1.0)))
    is_uniform = max_dev <= uniform_tol

    if verbose:
        print("Warp measure density diagnostics")
        print(f"  t_ref={t_ref}, n_points={len(s_vals)}")
        print(f"  z range: [{z_vals.min():.6e}, {z_vals.max():.6e}]")
        print(f"  measure_s range: [{measure_s.min():.6e}, {measure_s.max():.6e}]")
        print(
            f"  measure_z range (active): "
            f"[{measure_z_active.min():.6e}, {measure_z_active.max():.6e}]"
        )
        print(f"  active s interval: [{s_vals[active][0]:.6g}, {s_vals[active][-1]:.6g}]")
        print(f"  max|measure_z - 1| (active) = {max_dev:.6e}")
        print(f"  mean|measure_z - 1| (active) = {mean_dev:.6e}")
        print(f"  max|direct - via_inv| = {inv_agreement:.6e}")
        if not is_uniform:
            print(
                "  WARNING: pushed-forward measure deviates from uniform "
                f"(tol={uniform_tol:.1e})"
            )

    return WarpMeasureDensityResult(
        s_vals=s_vals,
        measure_s=measure_s,
        z_vals=z_active,
        measure_z=measure_z_active,
        max_pushforward_deviation_from_1=max_dev,
        mean_pushforward_deviation_from_1=mean_dev,
        z_min=float(z_active.min()),
        z_max=float(z_active.max()),
        is_uniform=is_uniform,
    )


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
    "WarpMeasureDensityResult",
    "build_interpolated_signal",
    "compute_inner_product",
    "compute_warp_measure_densities",
    "reconstruct_signal",
    "run_projection_reconstruction",
    "orthonormality_matrix_gauss",
    "orthonormality_summary",
]
