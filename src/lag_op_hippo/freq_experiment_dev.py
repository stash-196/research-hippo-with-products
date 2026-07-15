# %%
"""Interactive development workflow for the frequency-preserving Lag-SSM.

Artifacts are written to ``outputs/freq_dev/latest/<check_name>/`` when
``SAVE_ARTIFACTS=True`` so collaborators (and agents) can inspect figures
and metrics without relying on inline notebook output.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from scipy.integrate import quad

from lag_op_hippo.freq_dev_artifacts import (
    archive_latest_output,
    artifact_path,
    init_dev_output,
    save_metrics,
    save_run_config,
)
from lag_op_hippo.freq_diagnostics import (
    build_interpolated_signal,
    orthonormality_matrix_gauss,
    orthonormality_summary,
    run_projection_reconstruction,
)
from lag_op_hippo.lag_hippo import (
    LagHiPPO,
    lorenzsignal,
    phi,
    phi_deriv,
    sigma,
    sigma_deriv,
    sigma_inv,
    sigma_inv_deriv,
)
from lag_op_hippo.hippo_by_original_authors import HiPPO
from lag_op_hippo.lag_hippo_freq import (
    make_frequency_preserving_basis,
    make_frequency_preserving_model,
)
from lag_op_hippo.lag_hippo_translated import (
    compare_continuous_A_to_hippo,
    compare_discrete_A_to_hippo,
    make_fout_basis,
    make_fout_model,
    make_legt_basis,
    make_legt_model,
)
from lag_op_hippo.visualizers import (
    plot_basis_functions,
    plot_inner_product_matrix,
    plot_measure_density_warp,
    plot_reconstructed_signal,
    plot_roundtrip_error,
)

# --- Tunable parameters ---
ALPHA = 0.2
L_FOURIER = 1.0
EPS = 1e-8
N_FOURIER = 5
N_FOUT = 16
WINDOW_L = 1.0
DT = 0.001
T_LONG = 10.0
T_REF = T_LONG  # reference time t for warp/measure diagnostic plots
PAPER_READY_PLOTS = True
SAVE_ARTIFACTS = True
ARCHIVE_RUN = False
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "1") != "0"

OUT = init_dev_output(save_artifacts=SAVE_ARTIFACTS)
basis = make_frequency_preserving_basis(
    alpha=ALPHA,
    window_length=L_FOURIER,
    eps=EPS,
)

save_run_config(
    OUT,
    {
        "alpha": ALPHA,
        "window_length": L_FOURIER,
        "eps": EPS,
        "N_fourier": N_FOURIER,
        "dt": DT,
        "T_long": T_LONG,
        "save_artifacts": SAVE_ARTIFACTS,
    },
)

SIGMA_LABELS = [r"$\sigma_t'(s)$", r"$\sigma_t(s)$"]


def plot_sigma_warp_diagnostics(
    *,
    check: str,
    sigma_fn,
    sigma_deriv_fn,
    sigma_inv_fn,
    sigma_inv_deriv_fn,
    title_z: str,
    title_s: str,
    title_roundtrip: str,
    title_measure: str,
    warp_family: str,
    t_ref: float = 10.0,
):
    """Shared sigma-warp visuals: warp map, measure densities, and roundtrip."""
    vals = np.arange(0.0, t_ref, DT)
    sigmas_as_basis = np.column_stack(
        [sigma_deriv_fn(t_ref, vals), sigma_fn(t_ref, vals)]
    )

    plot_basis_functions(
        x=sigma_fn(t_ref, vals),
        Bases=sigmas_as_basis,
        ks=range(sigmas_as_basis.shape[1]),
        data_name="sigmas",
        domain_name="z",
        basis_type=warp_family,
        title=title_z,
        labels_list=SIGMA_LABELS,
        paper_ready=PAPER_READY_PLOTS,
        save_path=artifact_path(OUT, check, "sigma_vs_z.png"),
        show=SHOW_PLOTS,
    )
    plot_basis_functions(
        x=vals,
        Bases=sigmas_as_basis,
        ks=range(sigmas_as_basis.shape[1]),
        data_name="sigmas",
        domain_name="s",
        basis_type=warp_family,
        title=title_s,
        labels_list=SIGMA_LABELS,
        paper_ready=PAPER_READY_PLOTS,
        save_path=artifact_path(OUT, check, "sigma_vs_s.png"),
        show=SHOW_PLOTS,
    )

    s_test = np.linspace(1e-3, t_ref - 1e-3, 500)
    s_roundtrip = sigma_inv_fn(t_ref, sigma_fn(t_ref, s_test))
    roundtrip_err = np.abs(s_test - s_roundtrip)

    plot_roundtrip_error(
        s_test,
        roundtrip_err,
        domain_name="s",
        title=title_roundtrip,
        ylabel=r"$|\,s - \sigma_t^{-1}(\sigma_t(s))\,|$",
        label=r"$|\,s - \sigma_t^{-1}(\sigma_t(s))\,|$",
        paper_ready=PAPER_READY_PLOTS,
        save_path=artifact_path(OUT, check, "sigma_roundtrip_error.png"),
        show=SHOW_PLOTS,
    )

    measure_stats = plot_measure_density_warp(
        sigma_fn=sigma_fn,
        sigma_deriv_fn=sigma_deriv_fn,
        sigma_inv_fn=sigma_inv_fn,
        sigma_inv_deriv_fn=sigma_inv_deriv_fn,
        t_ref=t_ref,
        dt=DT,
        title=title_measure,
        paper_ready=PAPER_READY_PLOTS,
        save_path=artifact_path(OUT, check, "measure_density.png"),
        show=SHOW_PLOTS,
        verbose=True,
    )

    save_metrics(
        artifact_path(OUT, check, "metrics.json"),
        {
            "warp_family": warp_family,
            "t_ref": t_ref,
            "max_roundtrip_abs_error": float(np.max(roundtrip_err)),
            "mean_roundtrip_abs_error": float(np.mean(roundtrip_err)),
            **measure_stats,
        },
    )


def run_translated_ssm_validation(
    *,
    check: str,
    lag_model: LagHiPPO,
    hippo_method: str,
    basis,
    N: int,
    dt: float,
    T: float,
):
    """Compare LagOp translated SSM against original HiPPO reference."""
    hippo = HiPPO(method=hippo_method, N=N, dt=dt, T=T)
    vals = np.arange(0.0, T, dt)
    u = np.sin(2.0 * np.pi * 3.0 * vals)
    u_torch = torch.tensor(u, dtype=torch.float)

    c_hippo = hippo(u_torch)
    c_lag = lag_model(u_torch)
    recon_hippo = hippo.reconstruct(c_hippo[-1]).numpy()
    recon_lag = lag_model.reconstruct(c_lag[-1]).numpy()

    proj = run_projection_reconstruction(
        u_func=build_interpolated_signal(vals, u),
        s_vals=vals,
        phi_func=basis.phi,
        sigma_func=basis.sigma,
        sigma_inv_func=basis.sigma_inv,
        N=N,
        t=T,
        verbose=False,
    )
    inner = orthonormality_matrix_gauss(
        N=min(N, 8),
        phi_func=basis.phi,
        Z=(0.0, 1.0),
        quad_order=2000,
    )
    ortho = orthonormality_summary(inner, tolerance=1e-3)

    metrics = {
        "hippo_method": hippo_method,
        "N": N,
        "dt": dt,
        "T": T,
        "continuous_A_rel_error": compare_continuous_A_to_hippo(lag_model, hippo.A),
        "discrete_dA_rel_error": compare_discrete_A_to_hippo(lag_model, hippo.dA.numpy()),
        "recurrent_recon_mse_vs_hippo": float(np.mean((recon_hippo - recon_lag) ** 2)),
        "recurrent_recon_rel_error_vs_hippo": float(
            np.linalg.norm(recon_hippo - recon_lag) / np.linalg.norm(recon_hippo)
        ),
        "projection_recon_mse": proj.mse,
        "orthogonality_max_offdiag": ortho["max_offdiag"],
        "orthogonality_max_diag_deviation": ortho["max_diag_deviation"],
    }
    print(f"{check} validation ({hippo_method})")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    plot_reconstructed_signal(
        vals,
        u,
        [recon_hippo, recon_lag],
        data_name="",
        domain_name="s",
        title=f"LagOp vs HiPPO ({hippo_method})",
        paper_ready=PAPER_READY_PLOTS,
        recon_labels_list=["HiPPO reference", "LagOp translated"],
        recon_linestyle_list=["-", "-."],
        function_name="u",
        legend_loc="upper left",
        save_path=artifact_path(OUT, check, "reconstruction_vs_hippo.png"),
        show=SHOW_PLOTS,
    )
    save_metrics(artifact_path(OUT, check, "metrics.json"), metrics)


# %%
# Check 01: sigma warp and roundtrip (frequency-preserving)
plot_sigma_warp_diagnostics(
    check="01_sigma_warp",
    sigma_fn=basis.sigma,
    sigma_deriv_fn=basis.sigma_deriv,
    sigma_inv_fn=basis.sigma_inv,
    sigma_inv_deriv_fn=basis.sigma_inv_deriv,
    title_z=r"Freq-preserving $\sigma_t(s)$ on $[0,t]$ mapped to $z$",
    title_s=r"Freq-preserving $\sigma_t(s)$ on $[0,t]$",
    title_roundtrip=r"Freq-preserving $\sigma_t^{-1}(\sigma_t(s))$ roundtrip error",
    title_measure=r"Freq-preserving measure: $\omega_t(s)=\sigma_t'(s)$ and pushforward on $z$",
    warp_family="frequency-preserving",
)

# %%
# Check 01.1: sigma warp and roundtrip (LagOp HiPPO-LegS / exponential)
plot_sigma_warp_diagnostics(
    check="01_1_sigma_warp_lagop",
    sigma_fn=sigma,
    sigma_deriv_fn=sigma_deriv,
    sigma_inv_fn=sigma_inv,
    sigma_inv_deriv_fn=sigma_inv_deriv,
    title_z=r"LagOp HiPPO-LegS $\sigma_t(s)=\exp(s-t)$ on $[0,t]$ mapped to $z$",
    title_s=r"LagOp HiPPO-LegS $\sigma_t(s)=\exp(s-t)$ on $[0,t]$",
    title_roundtrip=r"LagOp HiPPO-LegS $\sigma_t^{-1}(\sigma_t(s))$ roundtrip error",
    title_measure=(
        r"LagOp HiPPO-LegS measure: $\omega_t(s)=e^{s-t}$ (decay on $s$), "
        r"uniform on $z$"
    ),
    warp_family="lagop-legs",
)

# %%
# Check 01.2: sigma warp and roundtrip (LagOp HiPPO-LegT / truncated translate)
legt_basis = make_legt_basis(window_length=WINDOW_L)
plot_sigma_warp_diagnostics(
    check="01_2_sigma_warp_legt",
    sigma_fn=legt_basis.sigma,
    sigma_deriv_fn=legt_basis.sigma_deriv,
    sigma_inv_fn=legt_basis.sigma_inv,
    sigma_inv_deriv_fn=legt_basis.sigma_inv_deriv,
    title_z=(
        rf"LagOp LegT at $t={T_REF}$: $\sigma_t(s)$ on $[0,t]$ mapped to $z$ "
        rf"(active on $[{T_REF - WINDOW_L}, {T_REF}]$, $L={WINDOW_L}$)"
    ),
    title_s=(
        rf"LagOp LegT at $t={T_REF}$: $\sigma_t(s)$ on $[0,t]$ "
        rf"(active on $[{T_REF - WINDOW_L}, {T_REF}]$)"
    ),
    title_roundtrip=rf"LagOp LegT at $t={T_REF}$: $\sigma_t^{{-1}}(\sigma_t(s))$ roundtrip error",
    title_measure=(
        rf"LagOp LegT at $t={T_REF}$: $\omega_t(s)=\sigma_t'(s)$ "
        rf"(zero on $[0,{T_REF - WINDOW_L})$, uniform on $[{T_REF - WINDOW_L},{T_REF}]$)"
    ),
    warp_family="lagop-legt",
    t_ref=T_REF,
)

# %%
# Check 01.3: sigma warp and roundtrip (LagOp HiPPO-FouT / truncated translate)
fout_basis = make_fout_basis(window_length=WINDOW_L)
plot_sigma_warp_diagnostics(
    check="01_3_sigma_warp_fout",
    sigma_fn=fout_basis.sigma,
    sigma_deriv_fn=fout_basis.sigma_deriv,
    sigma_inv_fn=fout_basis.sigma_inv,
    sigma_inv_deriv_fn=fout_basis.sigma_inv_deriv,
    title_z=(
        rf"LagOp FouT at $t={T_REF}$: $\sigma_t(s)$ on $[0,t]$ mapped to $z$ "
        rf"(active on $[{T_REF - WINDOW_L}, {T_REF}]$, $L={WINDOW_L}$)"
    ),
    title_s=(
        rf"LagOp FouT at $t={T_REF}$: $\sigma_t(s)$ on $[0,t]$ "
        rf"(active on $[{T_REF - WINDOW_L}, {T_REF}]$)"
    ),
    title_roundtrip=rf"LagOp FouT at $t={T_REF}$: $\sigma_t^{{-1}}(\sigma_t(s))$ roundtrip error",
    title_measure=(
        rf"LagOp FouT at $t={T_REF}$: $\omega_t(s)=\sigma_t'(s)$ "
        rf"(zero on $[0,{T_REF - WINDOW_L})$, uniform on $[{T_REF - WINDOW_L},{T_REF}]$)"
    ),
    warp_family="lagop-fout",
    t_ref=T_REF,
)

# %%
# Check 01.4: LagOp LegT / FouT validation vs original HiPPO
N_LEGT = 16
run_translated_ssm_validation(
    check="01_4_legt_vs_hippo",
    lag_model=make_legt_model(N=N_LEGT, dt=DT, T=WINDOW_L, window_length=WINDOW_L),
    hippo_method="legt",
    basis=legt_basis,
    N=N_LEGT,
    dt=DT,
    T=WINDOW_L,
)
run_translated_ssm_validation(
    check="01_5_fout_vs_hippo",
    lag_model=make_fout_model(N=N_FOUT, dt=DT, T=WINDOW_L, window_length=WINDOW_L),
    hippo_method="fourier",
    basis=fout_basis,
    N=N_FOUT,
    dt=DT,
    T=WINDOW_L,
)

# %%
# Check 02: recurrent reconstruction (freq-preserving vs standard)
check = "02_recurrent_reconstruction"
vals = np.arange(0.0, T_LONG, DT)
u = np.sin(2 * np.pi * 2 * vals)
u_torch = torch.tensor(u, dtype=torch.float)

model_fourier = make_frequency_preserving_model(
    N=N_FOURIER,
    dt=DT,
    T=T_LONG,
    alpha=ALPHA,
    window_length=L_FOURIER,
    eps=EPS,
    derive_A_from="infty",
    input_hold="zoh",
    quad_order=200 * N_FOURIER,
)
model_standard = LagHiPPO(
    N=N_FOURIER,
    phi_func=phi,
    phi_deriv_func=phi_deriv,
    sigma_func=sigma,
    sigma_deriv_func=sigma_deriv,
    sigma_inv_func=sigma_inv,
    sigma_inv_deriv_func=sigma_inv_deriv,
    derive_A_from="infty",
    input_hold="zoh",
    dt=DT,
    T=T_LONG,
)

c_fourier = model_fourier.forward(u_torch)
recon_fourier = model_fourier.reconstruct(c_fourier[-1]).numpy()
mse_fourier = float(np.mean((u - recon_fourier) ** 2))
print(f"MSE Fourier: {mse_fourier}")

c_standard = model_standard.forward(u_torch)
recon_standard = model_standard.reconstruct(c_standard[-1]).numpy()
mse_standard = float(np.mean((u - recon_standard) ** 2))
print(f"MSE Standard: {mse_standard}")

rel_diff = float(
    np.linalg.norm(recon_fourier - recon_standard)
    / np.linalg.norm(recon_standard)
)
print(f"Relative difference standard vs freq-preserving: {rel_diff:.6f}")

plot_basis_functions(
    x=vals,
    Bases=model_standard.eval_matrix.numpy()[::-1],
    ks=range(N_FOURIER),
    data_name="Basis Functions",
    domain_name="s",
    basis_type="frequency-preserving",
    title=r"Standard $\Psi_{0,t}$",
    paper_ready=PAPER_READY_PLOTS,
    save_path=artifact_path(OUT, check, "standard_basis_psi.png"),
    show=SHOW_PLOTS,
)
plot_basis_functions(
    x=sigma(vals, 0),
    Bases=model_standard.eval_matrix,
    ks=range(N_FOURIER),
    data_name="Basis Functions",
    domain_name="z",
    basis_type="frequency-preserving",
    title=r"Standard $\Phi_{0,t}$",
    paper_ready=PAPER_READY_PLOTS,
    save_path=artifact_path(OUT, check, "standard_basis_phi.png"),
    show=SHOW_PLOTS,
)
plot_basis_functions(
    x=vals,
    Bases=model_fourier.eval_matrix.numpy()[::-1],
    ks=range(N_FOURIER),
    data_name="Basis Functions",
    domain_name="s",
    basis_type="frequency-preserving",
    title=r"Freq-preserving $\Psi_{0,t}$",
    paper_ready=PAPER_READY_PLOTS,
    save_path=artifact_path(OUT, check, "freq_basis_psi.png"),
    show=SHOW_PLOTS,
)
plot_basis_functions(
    x=basis.sigma(vals, 0),
    Bases=model_fourier.eval_matrix,
    ks=range(N_FOURIER),
    data_name="Basis Functions",
    domain_name="z",
    basis_type="frequency-preserving",
    title=r"Freq-preserving $\Phi_{0,t}$",
    paper_ready=PAPER_READY_PLOTS,
    save_path=artifact_path(OUT, check, "freq_basis_phi.png"),
    show=SHOW_PLOTS,
)

plot_reconstructed_signal(
    vals,
    u,
    [
        recon_standard,
        recon_fourier,
        model_standard.measure[::-1] - 3.0,
        model_fourier.measure[::-1] - 3.0,
    ],
    data_name="",
    domain_name="s",
    title="Recurrent reconstruction",
    paper_ready=PAPER_READY_PLOTS,
    recon_labels_list=[
        "LagOp HiPPO-Standard",
        "LagOp HiPPO-Freq-Preserving",
        "Standard Measure",
        "Freq-Preserving Measure",
    ],
    recon_linestyle_list=["-", "-.", "--", "-"],
    function_name="u",
    legend_loc="upper left",
    xlim=(vals[0] - 1, vals[-1]),
    ylim=(-3.5, 3.5),
    save_path=artifact_path(OUT, check, "reconstruction_compare.png"),
    show=SHOW_PLOTS,
)

save_metrics(
    artifact_path(OUT, check, "metrics.json"),
    {
        "mse_fourier": mse_fourier,
        "mse_standard": mse_standard,
        "relative_reconstruction_diff": rel_diff,
        "recon_mse_between_models": float(np.mean((recon_standard - recon_fourier) ** 2)),
    },
)

# %%
# Check 03: direct projection-reconstruction (non-recurrent)
check = "03_projection_reconstruction"
T = 1.0
dt = 0.001
proj_vals = np.arange(0.0, T, dt)
u_lorenz = lorenzsignal(T, dt, initial_state=[1.0, 1.0, 8.0])[:, 0]
u_lorenz = (u_lorenz - np.mean(u_lorenz)) / np.std(u_lorenz)
s_vals = np.linspace(0.0, T, int(T / dt) + 1)
u_extended = np.interp(s_vals, proj_vals, u_lorenz)
u_func = build_interpolated_signal(s_vals, u_extended)

result_exp = run_projection_reconstruction(
    u_func=u_func,
    s_vals=s_vals,
    phi_func=phi,
    sigma_func=sigma,
    sigma_inv_func=sigma_inv,
    N=64,
    t=T,
    verbose=True,
)
result_freq = run_projection_reconstruction(
    u_func=u_func,
    s_vals=s_vals,
    phi_func=basis.phi,
    sigma_func=basis.sigma,
    sigma_inv_func=basis.sigma_inv,
    N=64,
    t=T,
    verbose=True,
)

plot_reconstructed_signal(
    s_vals,
    u_extended,
    [result_exp.reconstruction, result_freq.reconstruction],
    data_name="",
    domain_name="s",
    title="Projection-reconstruction",
    paper_ready=PAPER_READY_PLOTS,
    recon_labels_list=["Exponential", "Freq-Preserving"],
    recon_linestyle_list=["-", "-"],
    function_name="u",
    legend_loc="upper left",
    xlim=(s_vals[0] - 0.1, s_vals[-1]),
    save_path=artifact_path(OUT, check, "projection_reconstruction.png"),
    show=SHOW_PLOTS,
)

save_metrics(
    artifact_path(OUT, check, "metrics.json"),
    {
        "mse_exponential": result_exp.mse,
        "mse_freq_preserving": result_freq.mse,
        "runtime_s_exponential": result_exp.runtime_s,
        "runtime_s_freq_preserving": result_freq.runtime_s,
    },
)

# %%
# Check 04: orthonormality via adaptive quadrature (per-window intervals)
check = "04_orthonormality_quad"


def test_orthonormality_quad(phi_func, N, Z=(0.0, 1.0), atol=1e-8):
    failures = []
    for n in range(N):
        for m in range(N):

            def integrand(z, n=n, m=m):
                return phi_func(n, z) * phi_func(m, z)

            result, _ = quad(integrand, Z[0], Z[1], epsabs=1e-12, epsrel=1e-12, limit=10000)
            if n != m and not np.isclose(result, 0.0, atol=atol):
                failures.append({"n": n, "m": m, "inner_product": float(result)})
    return failures


N_FOURIER_TEST = 5
quad_results = []
print("Check 04: orthonormality (adaptive quadrature)")
for N in range(N_FOURIER_TEST + 1):
    z_min = basis.sigma(10.0, 10.0 - N * L_FOURIER)
    z_max = basis.sigma(10.0, 10.0 - (N - 1) * L_FOURIER)
    failures = test_orthonormality_quad(basis.phi, N, Z=(z_min, z_max))
    status = "pass" if not failures else "fail"
    print(f"  window N={N} over [{z_min:.3g}, {z_max:.3g}]: {status}")
    quad_results.append(
        {
            "N": N,
            "z_min": float(z_min),
            "z_max": float(z_max),
            "status": status,
            "failures": failures,
        }
    )

save_metrics(artifact_path(OUT, check, "metrics.json"), {"per_window": quad_results})

# %%
# Check 05: orthonormality via Gauss quadrature + heatmaps
check = "05_orthonormality_gauss"
N_FOURIER_TEST = 20
quad_order = 100 * N_FOURIER_TEST
gauss_results = []

print("Check 05: orthonormality (Gauss quadrature)")
for N in range(2, N_FOURIER_TEST + 1):
    z_min = basis.sigma(10.0, 10.0 - N * L_FOURIER)
    inner_mat = orthonormality_matrix_gauss(
        N=N,
        phi_func=basis.phi,
        Z=(z_min, 1.0),
        quad_order=quad_order,
    )
    summary = orthonormality_summary(inner_mat, tolerance=1e-4)
    print(
        f"  N={N} over [{z_min:.3g}, 1]: "
        f"max_offdiag={summary['max_offdiag']:.3e}, "
        f"max_diag_dev={summary['max_diag_deviation']:.3e}"
    )
    gauss_results.append(
        {
            "N": N,
            "z_min": float(z_min),
            "z_max": 1.0,
            **summary,
        }
    )
    if N == N_FOURIER_TEST:
        plot_inner_product_matrix(
            inner_mat,
            title=f"Inner product matrix (N={N})",
            paper_ready=PAPER_READY_PLOTS,
            save_path=artifact_path(OUT, check, f"inner_matrix_N{N}.png"),
            show=SHOW_PLOTS,
        )

inner_full = orthonormality_matrix_gauss(
    N=5,
    phi_func=basis.phi,
    Z=(0.0, 1.0),
    quad_order=quad_order * 20,
)
summary_full = orthonormality_summary(inner_full, tolerance=1e-5)
plot_inner_product_matrix(
    inner_full,
    title="Inner product matrix over [0, 1], N=5",
    paper_ready=PAPER_READY_PLOTS,
    save_path=artifact_path(OUT, check, "inner_matrix_full_0_1_N5.png"),
    show=SHOW_PLOTS,
)

save_metrics(
    artifact_path(OUT, check, "metrics.json"),
    {
        "quad_order": quad_order,
        "per_N": gauss_results,
        "full_interval_0_1_N5": summary_full,
    },
)

# %%
if SAVE_ARTIFACTS and ARCHIVE_RUN:
    archived = archive_latest_output(OUT)
    if archived is not None:
        print(f"Archived run to {archived}")

if OUT is not None:
    print(f"Artifacts saved under {OUT}")
# %%