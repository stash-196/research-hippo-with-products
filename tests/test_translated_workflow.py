import numpy as np
import torch

from lag_op_hippo.freq_diagnostics import (
    build_interpolated_signal,
    compute_warp_measure_densities,
    orthonormality_matrix_gauss,
    orthonormality_summary,
    run_projection_reconstruction,
)
from lag_op_hippo.hippo_by_original_authors import HiPPO
from lag_op_hippo.lag_hippo_translated import (
    compare_discrete_A_to_hippo,
    make_fout_basis,
    make_fout_model,
    make_legt_basis,
    make_legt_model,
)


def test_truncated_translate_pushforward_is_uniform():
    basis = make_legt_basis(window_length=1.0)
    result = compute_warp_measure_densities(
        sigma_fn=basis.sigma,
        sigma_deriv_fn=basis.sigma_deriv,
        sigma_inv_fn=basis.sigma_inv,
        sigma_inv_deriv_fn=basis.sigma_inv_deriv,
        t_ref=1.0,
        dt=0.01,
    )
    assert result.is_uniform
    assert result.max_pushforward_deviation_from_1 < 1e-10


def test_legt_basis_is_orthogonal_on_unit_interval():
    basis = make_legt_basis(window_length=1.0)
    inner = orthonormality_matrix_gauss(
        N=8,
        phi_func=basis.phi,
        Z=(0.0, 1.0),
        quad_order=2000,
    )
    summary = orthonormality_summary(inner, tolerance=1e-3)
    assert summary["max_offdiag"] < 1e-3


def test_fout_basis_is_orthogonal_on_unit_interval():
    basis = make_fout_basis(window_length=1.0)
    inner = orthonormality_matrix_gauss(
        N=8,
        phi_func=basis.phi,
        Z=(0.0, 1.0),
        quad_order=2000,
    )
    summary = orthonormality_summary(inner, tolerance=1e-3)
    assert summary["max_offdiag"] < 1e-3


def test_legt_projection_reconstruction_is_accurate():
    basis = make_legt_basis(window_length=1.0)
    T, dt = 1.0, 0.001
    s_vals = np.linspace(0.0, T, int(T / dt) + 1)
    u_vals = np.sin(2.0 * np.pi * 3.0 * s_vals)
    u_func = build_interpolated_signal(s_vals, u_vals)
    result = run_projection_reconstruction(
        u_func=u_func,
        s_vals=s_vals,
        phi_func=basis.phi,
        sigma_func=basis.sigma,
        sigma_inv_func=basis.sigma_inv,
        N=16,
        t=T,
        verbose=False,
    )
    assert result.mse < 1e-4


def test_fout_projection_reconstruction_is_reasonable():
    basis = make_fout_basis(window_length=1.0)
    T, dt = 1.0, 0.001
    s_vals = np.linspace(0.0, T, int(T / dt) + 1)
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
        verbose=False,
    )
    assert np.isfinite(result.mse)
    assert result.mse < 0.5


def test_legt_lagop_model_forward_is_finite():
    T, dt = 1.0, 0.01
    vals = np.arange(0.0, T, dt)
    u = torch.tensor(np.sin(2.0 * np.pi * 2.0 * vals), dtype=torch.float32)
    model = make_legt_model(N=16, dt=dt, T=T)
    coeffs = model.forward(u)
    recon = model.reconstruct(coeffs[-1]).detach().cpu().numpy()
    assert np.all(np.isfinite(coeffs.detach().cpu().numpy()))
    assert np.all(np.isfinite(recon))


def test_fout_lagop_model_forward_is_finite():
    T, dt = 1.0, 0.01
    vals = np.arange(0.0, T, dt)
    u = torch.tensor(np.sin(2.0 * np.pi * 2.0 * vals), dtype=torch.float32)
    model = make_fout_model(N=16, dt=dt, T=T)
    coeffs = model.forward(u)
    recon = model.reconstruct(coeffs[-1]).detach().cpu().numpy()
    assert np.all(np.isfinite(coeffs.detach().cpu().numpy()))
    assert np.all(np.isfinite(recon))


def test_legt_discrete_A_is_within_expected_tolerance_of_hippo():
    """Full discrete-A equality is still open; track current best-effort gap."""
    N, dt, T = 16, 0.01, 1.0
    hippo = HiPPO(method="legt", N=N, dt=dt, T=T)
    lag = make_legt_model(N=N, dt=dt, T=T, quad_order=200 * N)
    rel = compare_discrete_A_to_hippo(lag, hippo.dA.numpy())
    assert rel < 0.75