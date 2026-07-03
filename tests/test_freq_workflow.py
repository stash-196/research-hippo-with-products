import numpy as np
import torch

from lag_op_hippo.freq_diagnostics import (
    build_interpolated_signal,
    orthonormality_matrix_gauss,
    run_projection_reconstruction,
)
from lag_op_hippo.lag_hippo_freq import (
    make_frequency_preserving_basis,
    make_frequency_preserving_model,
)


def test_sigma_inverse_roundtrip_precision():
    basis = make_frequency_preserving_basis(alpha=0.2, window_length=1.0)

    t = 5.0
    s_vals = np.linspace(1e-3, t - 1e-3, 500)
    z_vals = basis.sigma(t, s_vals)
    roundtrip_s = basis.sigma_inv(t, z_vals)

    max_abs_err = float(np.max(np.abs(s_vals - roundtrip_s)))
    assert max_abs_err < 1e-10


def test_sigma_and_inverse_derivatives_are_finite_and_valid():
    basis = make_frequency_preserving_basis(alpha=0.2, window_length=1.0)

    t = 3.0
    s_vals = np.linspace(0.0, t, 300)
    z_vals = basis.sigma(t, s_vals)
    assert np.all(np.isfinite(z_vals))
    assert np.all((z_vals >= 0.0) & (z_vals <= 1.0))

    z_grid = np.linspace(1e-8, 1.0, 400)
    inv_vals = basis.sigma_inv(t, z_grid)
    inv_deriv_vals = basis.sigma_inv_deriv(t, z_grid)

    assert np.all(np.isfinite(inv_vals))
    assert np.all(np.isfinite(inv_deriv_vals))
    assert np.all(inv_deriv_vals > 0.0)


def test_basis_is_nearly_orthogonal_on_off_diagonals():
    basis = make_frequency_preserving_basis(alpha=0.2, window_length=1.0)

    inner = orthonormality_matrix_gauss(
        N=6,
        phi_func=basis.phi,
        Z=(0.0, 1.0),
        quad_order=2000,
    )

    diag = np.diag(inner)
    offdiag = inner - np.diag(diag)
    max_offdiag = float(np.max(np.abs(offdiag)))

    # This basis is not perfectly unit-normalized over [0,1],
    # but off-diagonal terms should remain very small.
    assert max_offdiag < 1e-3
    assert np.all(diag > 0.0)
    assert float(np.max(diag) / np.min(diag)) < 3.0


def test_projection_reconstruction_on_sinusoid_is_finite_and_reasonable():
    basis = make_frequency_preserving_basis(alpha=0.2, window_length=1.0)

    T = 1.0
    dt = 0.01
    s_vals = np.arange(0.0, T + dt / 2, dt)
    u_vals = np.sin(2.0 * np.pi * 2.0 * s_vals)
    u_func = build_interpolated_signal(s_vals, u_vals)

    result = run_projection_reconstruction(
        u_func=u_func,
        s_vals=s_vals,
        phi_func=basis.phi,
        sigma_func=basis.sigma,
        sigma_inv_func=basis.sigma_inv,
        N=16,
        t=T,
        quad_order=128,
        verbose=False,
    )

    assert np.isfinite(result.mse)
    assert result.mse < 0.2
    assert np.all(np.isfinite(result.coefficients))
    assert np.all(np.isfinite(result.reconstruction))


def test_frequency_model_forward_reconstruct_is_finite_small_config():
    T = 0.5
    dt = 0.01
    vals = np.arange(0.0, T, dt)
    u_vals = np.sin(2.0 * np.pi * 3.0 * vals).astype(np.float32)
    u = torch.tensor(u_vals, dtype=torch.float32)

    model = make_frequency_preserving_model(
        N=8,
        dt=dt,
        T=T,
        alpha=0.2,
        window_length=1.0,
        # This config is stable for quick sanity tests.
        derive_A_from="gen",
        input_hold="delta",
        quad_order=64,
    )

    coeffs = model.forward(u)
    recon = model.reconstruct(coeffs[-1]).detach().cpu().numpy()

    coeffs_np = coeffs.detach().cpu().numpy()
    assert coeffs_np.shape[0] == len(vals)
    assert recon.shape[0] == len(vals)
    assert np.all(np.isfinite(coeffs_np))
    assert np.all(np.isfinite(recon))
