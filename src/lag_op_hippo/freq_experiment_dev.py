# %%
"""Interactive development workflow for the frequency-preserving Lag-SSM.

Edit the **I/O toggles** block below to control whether figures are shown
and/or saved. Artifacts land under ``outputs/freq_dev/latest/<check_name>/``
when saving is enabled.
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
    compute_A_infty,
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
    plot_eval_matrix_columns,
    plot_inner_product_matrix,
    plot_measure_density_warp,
    plot_measure_vs_lookback,
    plot_reconstructed_signal,
    plot_roundtrip_error,
    plot_sigma_warp_pair,
)

# --- Tunable parameters ---
ALPHA = 0.2
L_FOURIER = 1.0
EPS = 1e-8
N_FOURIER = 5
N_FOUT = 16
N_LEGS = 16
N_LEGT = 16
N_BASIS_SHOW = 6  # how many eval-matrix columns to draw in geometry plots
N_DOMAIN_EXPANSION = 64  # high-order mode for Fig. 3-style lag visualization
WINDOW_L = 1.0
DT = 0.001
T_LONG = 10.0
T_REF = T_LONG  # reference time t for warp/measure diagnostic plots
T_DOMAIN = 3.0  # horizon for domain-expansion (paper Fig. 3 style)
DOMAIN_ZOOM = 20  # samples near the present for lag-shift zoom
PAPER_READY_PLOTS = True

# =============================================================================
# I/O toggles — edit these (env vars override the defaults if set)
# =============================================================================
# SHOW_PLOTS: open / display each figure in a window (or notebook cell).
# SAVE_PLOTS: write PNGs under outputs/freq_dev/latest/<check>/.
# SAVE_METRICS: write metrics.json / run_config.json alongside figures.
# ARCHIVE_RUN: copy latest/ → runs/<timestamp>/ after the full script finishes.
_env_bool = lambda key, default: (os.environ.get(key, "1" if default else "0") != "0")
SHOW_PLOTS = _env_bool("SHOW_PLOTS", True)
SAVE_PLOTS = _env_bool("SAVE_PLOTS", True)
SAVE_METRICS = _env_bool("SAVE_METRICS", True)
# Back-compat alias used by init_dev_output / archive
SAVE_ARTIFACTS = SAVE_PLOTS or SAVE_METRICS
ARCHIVE_RUN = _env_bool("ARCHIVE_RUN", True)

OUT = init_dev_output(save_artifacts=SAVE_ARTIFACTS)


def plot_io(check: str, filename: str) -> dict:
    """Standard ``show`` / ``save`` / ``save_path`` kwargs for visualizers.

    Respects the module-level ``SHOW_PLOTS`` and ``SAVE_PLOTS`` toggles so
    call sites stay short::

        plot_reconstructed_signal(..., **plot_io(check, "foo.png"), paper_ready=...)
    """
    path = artifact_path(OUT, check, filename) if SAVE_PLOTS else None
    return {"show": SHOW_PLOTS, "save": SAVE_PLOTS, "save_path": path}


def maybe_save_metrics(check: str, filename: str, metrics: dict) -> None:
    """Write metrics JSON only when ``SAVE_METRICS`` is enabled."""
    if not SAVE_METRICS:
        return
    save_metrics(artifact_path(OUT, check, filename), metrics)


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
        "show_plots": SHOW_PLOTS,
        "save_plots": SAVE_PLOTS,
        "save_metrics": SAVE_METRICS,
        "save_artifacts": SAVE_ARTIFACTS,
    },
)

SIGMA_LABELS = [r"$\sigma_t'(s)$", r"$\sigma_t(s)$"]


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_model_geometry(
    *,
    check: str,
    lookback,
    measure,
    eval_matrix,
    model_label: str,
    file_prefix: str,
    num_basis: int | None = None,
    plot_measure: bool = True,
    plot_eval: bool = True,
):
    """Plot measure vs lookback and/or eval-matrix columns for one model."""
    lookback = _as_numpy(lookback)
    measure = _as_numpy(measure)
    eval_matrix = _as_numpy(eval_matrix)
    n_show = (
        num_basis if num_basis is not None else min(N_BASIS_SHOW, eval_matrix.shape[1])
    )

    if plot_measure:
        plot_measure_vs_lookback(
            lookback,
            measure,
            labels=[model_label],
            title=rf"{model_label}: measure vs lookback",
            paper_ready=PAPER_READY_PLOTS,
            **plot_io(check, f"{file_prefix}_measure_vs_lookback.png"),
        )
    if plot_eval:
        plot_eval_matrix_columns(
            lookback,
            eval_matrix,
            title=rf"{model_label}: eval-matrix columns (first {n_show})",
            basis_type=file_prefix,
            num_functions_front=n_show,
            paper_ready=PAPER_READY_PLOTS,
            **plot_io(check, f"{file_prefix}_eval_matrix.png"),
        )


def plot_hippo_lagop_geometry_comparison(
    *,
    check: str,
    hippo_model: HiPPO | None,
    lag_model: LagHiPPO,
    family_label: str,
):
    """Measure + eval-matrix visuals for HiPPO (optional) and LagOp.

    Original HiPPO stores lookback on ``vals`` with present at ``tau=0``.
    LagOp uses the same convention for ``vals`` / ``measure`` / ``eval_matrix``.

    When both models share a lookback grid, measures are shown only as an
    overlay (individual measure plots would be duplicates). Eval matrices stay
    separate (no overlay). If there is no HiPPO counterpart (e.g. freq-preserving),
    LagOp measure is plotted individually.
    """
    lag_lookback = _as_numpy(lag_model.vals)
    lag_measure = _as_numpy(lag_model.measure)
    lag_eval = _as_numpy(lag_model.eval_matrix)

    if hippo_model is None:
        # No comparison target: keep individual LagOp measure + eval.
        plot_model_geometry(
            check=check,
            lookback=lag_lookback,
            measure=lag_measure,
            eval_matrix=lag_eval,
            model_label=f"LagOp {family_label}",
            file_prefix="lagop",
            plot_measure=True,
            plot_eval=True,
        )
        return

    hippo_lookback = _as_numpy(hippo_model.vals)
    hippo_measure = _as_numpy(hippo_model.measure)
    hippo_eval = _as_numpy(hippo_model.eval_matrix)

    can_overlay_measure = hippo_lookback.shape == lag_lookback.shape and np.allclose(
        hippo_lookback, lag_lookback
    )

    # Eval matrices always plotted per model (not overlaid).
    # Measures: overlay only when comparable; otherwise individual plots.
    plot_model_geometry(
        check=check,
        lookback=lag_lookback,
        measure=lag_measure,
        eval_matrix=lag_eval,
        model_label=f"LagOp {family_label}",
        file_prefix="lagop",
        plot_measure=not can_overlay_measure,
        plot_eval=True,
    )
    plot_model_geometry(
        check=check,
        lookback=hippo_lookback,
        measure=hippo_measure,
        eval_matrix=hippo_eval,
        model_label=f"HiPPO {family_label}",
        file_prefix="hippo",
        plot_measure=not can_overlay_measure,
        plot_eval=True,
    )

    if can_overlay_measure:
        plot_measure_vs_lookback(
            hippo_lookback,
            [hippo_measure, lag_measure],
            labels=[f"HiPPO {family_label}", f"LagOp {family_label}"],
            title=rf"{family_label}: measure vs lookback (HiPPO vs LagOp)",
            paper_ready=PAPER_READY_PLOTS,
            **plot_io(check, "measure_vs_lookback_overlay.png"),
        )


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
    sigma_on_s = sigma_fn(t_ref, vals)
    sigma_deriv_on_s = sigma_deriv_fn(t_ref, vals)

    # Single side-by-side figure: σ vs s | σ vs z  (replaces separate PNGs)
    plot_sigma_warp_pair(
        s_vals=vals,
        z_vals=sigma_on_s,
        sigma_vals=sigma_on_s,
        sigma_deriv_vals=sigma_deriv_on_s,
        title_s=title_s,
        title_z=title_z,
        labels=SIGMA_LABELS,
        paper_ready=PAPER_READY_PLOTS,
        **plot_io(check, "sigma_maps.png"),
    )

    s_test = np.linspace(1e-3, t_ref - 1e-3, 500)
    s_roundtrip = sigma_inv_fn(t_ref, sigma_fn(t_ref, s_test))
    roundtrip_err = np.abs(s_test - s_roundtrip)

    plot_roundtrip_error(
        s_test,
        roundtrip_err,
        s_roundtrip=s_roundtrip,
        domain_name="s",
        title=title_roundtrip,
        ylabel=r"$|\,s - \sigma_t^{-1}(\sigma_t(s))\,|$",
        label=r"$|\,s - \sigma_t^{-1}(\sigma_t(s))\,|$",
        paper_ready=PAPER_READY_PLOTS,
        **plot_io(check, "sigma_roundtrip_error.png"),
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
        **plot_io(check, "measure_density.png"),
        verbose=True,
    )

    maybe_save_metrics(
        check,
        "metrics.json",
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
    """Compare LagOp SSM (LegS / LegT / FouT) against original HiPPO reference."""
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
        "discrete_dA_rel_error": compare_discrete_A_to_hippo(
            lag_model, hippo.dA.numpy()
        ),
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
        recon_labels_list=["HiPPO reference", "LagOp"],
        recon_linestyle_list=["-", "-."],
        function_name="u",
        legend_loc="upper left",
        **plot_io(check, "reconstruction_vs_hippo.png"),
    )
    maybe_save_metrics(check, "metrics.json", metrics)


def run_domain_expansion_visualization(
    *,
    check: str,
    family_label: str,
    hippo_method: str,
    N: int,
    dt: float,
    T: float,
    phi_func,
    phi_deriv_func,
    sigma_func,
    sigma_deriv_func,
    sigma_inv_func,
    sigma_inv_deriv_func,
    mode: str = "scale",
    window: float = 1.0,
    a_tilt: str = "legs",
    example_n: int | None = None,
    zoom: int = DOMAIN_ZOOM,
    file_prefix: str | None = None,
):
    """Paper Fig. 3-style lag / domain-expansion of one high-order basis mode.

    Ground truth is the unshifted mode at horizon ``T``.  Forward domain
    expansion is shown two ways that should agree when LagOp recovers HiPPO:

    - solid: HiPPO reference ``dA`` applied to a one-hot coefficient vector
    - dashed: LagOp lag-operator forward map from ``A_∞`` (with LegS tilt
      correction when ``a_tilt='legs'``)

    Backward lag from ``A_∞`` is also shown for geometry (compress toward
    older history).
    """
    prefix = file_prefix or family_label.lower().replace(" ", "_")
    if example_n is None:
        example_n = N - 1
    mode_idx = max(example_n - 1, 0)  # demo.py one-hot index convention

    # LagOp models: eval_matrix / reconstruct only (lag maps from A_∞ below).
    # Avoid derive_A_from="infty" here: FouT's null sin(0) mode makes A_∞ singular.
    lag_t = LagHiPPO(
        N=N,
        phi_func=phi_func,
        phi_deriv_func=phi_deriv_func,
        sigma_func=sigma_func,
        sigma_deriv_func=sigma_deriv_func,
        sigma_inv_func=sigma_inv_func,
        sigma_inv_deriv_func=sigma_inv_deriv_func,
        dt=dt,
        T=T,
        derive_A_from="gen",
        mode=mode,
        window=window,
        a_tilt=a_tilt,
    )
    lag_t_plus = LagHiPPO(
        N=N,
        phi_func=phi_func,
        phi_deriv_func=phi_deriv_func,
        sigma_func=sigma_func,
        sigma_deriv_func=sigma_deriv_func,
        sigma_inv_func=sigma_inv_func,
        sigma_inv_deriv_func=sigma_inv_deriv_func,
        dt=dt,
        T=T + dt,
        derive_A_from="gen",
        mode=mode,
        window=window,
        a_tilt=a_tilt,
    )
    # Original HiPPO reference (Gu): discrete dA is the one-step state map.
    hippo_t = HiPPO(method=hippo_method, N=N, dt=dt, T=T)
    hippo_t_plus = HiPPO(method=hippo_method, N=N, dt=dt, T=T + dt)

    # Match eval_matrix dtype (often float64 from NumPy → torch.Tensor).
    dtype = lag_t.eval_matrix.dtype

    coeff_one_hot = torch.zeros(N, dtype=dtype)
    coeff_one_hot[mode_idx] = 1.0

    A_infty = compute_A_infty(
        N, phi_func, sigma_func, sigma_inv_func, dt=dt, verbose=False
    )
    # FouT null mode can make A_∞ singular → pinv for the forward map.
    A_infty_T_inv = np.linalg.pinv(A_infty.T)
    if a_tilt == "legs":
        # HiPPO-LegS stabilization correction (paper / demo.py)
        A_lag_backward = np.exp(dt) * A_infty
        A_lag_forward = A_infty_T_inv * np.exp(-dt)
    else:
        A_lag_backward = A_infty
        A_lag_forward = A_infty_T_inv

    dA_hippo = hippo_t.dA.detach().cpu().numpy().astype(np.float64)
    # Optional HiPPO backward: free evolution inverse (best-effort; may be ill-conditioned)
    dA_hippo_backward = np.linalg.pinv(dA_hippo)

    A_lag_forward_t = torch.as_tensor(A_lag_forward, dtype=dtype)
    A_lag_backward_t = torch.as_tensor(A_lag_backward, dtype=dtype)
    dA_hippo_t = hippo_t.dA.detach().cpu().to(dtype=dtype)
    dA_hippo_backward_t = torch.as_tensor(dA_hippo_backward, dtype=dtype)
    nan0 = torch.tensor([np.nan], dtype=dtype)

    # --- Ground truth: unshifted mode at t (LagOp and HiPPO should match) ---
    gt_lag = torch.cat((nan0, lag_t.eval_matrix @ coeff_one_hot)).flip(-1)
    gt_hippo = torch.cat(
        (nan0, hippo_t.eval_matrix.to(dtype=dtype) @ coeff_one_hot)
    ).flip(-1)

    # --- Forward: one free discrete step, then reconstruct on [0, T+dt] ---
    fwd_lag = lag_t_plus.reconstruct(A_lag_forward_t @ coeff_one_hot)
    fwd_hippo = hippo_t_plus.reconstruct(dA_hippo_t @ coeff_one_hot)

    # --- Backward lag (geometry); align to T+dt grid like demo.py ---
    bwd_lag = lag_t.eval_matrix @ (A_lag_backward_t @ coeff_one_hot)
    bwd_lag = bwd_lag.clone()
    bwd_lag[0] = np.nan
    bwd_lag = torch.cat((nan0, bwd_lag)).flip(-1)

    bwd_hippo = hippo_t.eval_matrix.to(dtype=dtype) @ (
        dA_hippo_backward_t @ coeff_one_hot
    )
    bwd_hippo = bwd_hippo.clone()
    bwd_hippo[0] = np.nan
    bwd_hippo = torch.cat((nan0, bwd_hippo)).flip(-1)

    vals_t_plus = torch.arange(0.0, T + dt, dt, dtype=dtype)
    zoom_n = min(zoom, len(vals_t_plus))
    sl = slice(-zoom_n, None)

    # Primary figure: GT + forward pair (HiPPO solid, LagOp dashed) + lag backward pair
    # Legend on the left (2 columns); wide canvas so the curve is not squashed.
    plot_reconstructed_signal(
        vals_t_plus[sl],
        gt_lag[sl],
        [
            fwd_hippo[sl],
            fwd_lag[sl],
            bwd_hippo[sl],
            bwd_lag[sl],
            gt_hippo[sl],
        ],
        data_name="",  # avoid redundant "… Basis" suffix on every legend entry
        function_name=rf"$\phi_{{{example_n}}}$",
        domain_name="s",
        title=rf"{family_label}: domain expansion (GT + HiPPO $dA$ vs LagOp $A_\infty$)",
        paper_ready=PAPER_READY_PLOTS,
        recon_labels_list=[
            r"HiPPO $dA$ forward",
            r"LagOp $A_\infty$ forward",
            r"HiPPO $dA^{-1}$ backward",
            r"LagOp $A_\infty$ backward",
            r"HiPPO GT (unshifted)",
        ],
        recon_linestyle_list=["-", "--", "-", "--", ":"],
        recon_color_list=[
            "tab:blue",  # HiPPO forward (solid)
            "tab:red",  # LagOp forward (dashed)
            "tab:orange",  # HiPPO backward (solid)
            "tab:purple",  # LagOp backward (dashed)
            "tab:green",  # HiPPO GT
        ],
        legend_outside="right",
        legend_ncols=1,
        legend_fontsize=11,
        figsize=(12.5, 5.0),
        **plot_io(check, f"{prefix}_domain_expansion.png"),
    )

    # Relative errors where finite
    def _rel_err(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        mask = np.isfinite(a) & np.isfinite(b)
        if not np.any(mask) or np.linalg.norm(b[mask]) == 0:
            return float("nan")
        return float(np.linalg.norm(a[mask] - b[mask]) / np.linalg.norm(b[mask]))

    metrics = {
        "family": family_label,
        "hippo_method": hippo_method,
        "N": N,
        "dt": dt,
        "T": T,
        "example_n": example_n,
        "mode_index": mode_idx,
        "a_tilt": a_tilt,
        "mode": mode,
        "zoom": zoom_n,
        "A_infty_fro": float(np.linalg.norm(A_infty, ord="fro")),
        "A_lag_forward_fro": float(np.linalg.norm(A_lag_forward, ord="fro")),
        "dA_hippo_fro": float(np.linalg.norm(dA_hippo, ord="fro")),
        "rel_err_dA_vs_A_lag_forward": float(
            np.linalg.norm(dA_hippo - A_lag_forward, ord="fro")
            / max(np.linalg.norm(dA_hippo, ord="fro"), 1e-12)
        ),
        "rel_err_forward_waveform_hippo_vs_lag": _rel_err(fwd_hippo, fwd_lag),
        "rel_err_gt_hippo_vs_lag": _rel_err(gt_hippo, gt_lag),
        "rel_err_backward_waveform_hippo_vs_lag": _rel_err(bwd_hippo, bwd_lag),
    }
    print(f"{check} domain expansion ({family_label})")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    maybe_save_metrics(check, f"{prefix}_metrics.json", metrics)
    return metrics


# %%
# Check 01.0: sigma warp and roundtrip (frequency-preserving)
# No original HiPPO counterpart — LagOp geometry only.
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
freq_model = make_frequency_preserving_model(
    N=N_FOURIER,
    dt=DT,
    T=T_LONG,
    alpha=ALPHA,
    window_length=L_FOURIER,
    eps=EPS,
)
plot_hippo_lagop_geometry_comparison(
    check="01_sigma_warp",
    hippo_model=None,
    lag_model=freq_model,
    family_label="Freq-preserving",
)


# Check 01.1: sigma warp + measure/eval geometry (LegS: HiPPO vs LagOp)
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
hippo_legs = HiPPO(method="legs", N=N_LEGS, dt=DT, T=T_LONG)
lag_legs = LagHiPPO(
    N=N_LEGS,
    phi_func=phi,
    phi_deriv_func=phi_deriv,
    sigma_func=sigma,
    sigma_deriv_func=sigma_deriv,
    sigma_inv_func=sigma_inv,
    sigma_inv_deriv_func=sigma_inv_deriv,
    dt=DT,
    T=T_LONG,
)
plot_hippo_lagop_geometry_comparison(
    check="01_1_sigma_warp_lagop",
    hippo_model=hippo_legs,
    lag_model=lag_legs,
    family_label="LegS",
)


# Check 01.2: sigma warp + measure/eval geometry (LegT: HiPPO vs LagOp)
legt_basis = make_legt_basis(window_length=WINDOW_L)
plot_sigma_warp_diagnostics(
    check="01_2_sigma_warp_legt",
    sigma_fn=legt_basis.sigma,
    sigma_deriv_fn=legt_basis.sigma_deriv,
    sigma_inv_fn=legt_basis.sigma_inv,
    sigma_inv_deriv_fn=legt_basis.sigma_inv_deriv,
    title_z=(
        rf"LagOp LegT at $t={T_REF}$:"
        "\n"
        rf"$\sigma_t(s)$ on $[0,t]\to z$"
        "\n"
        rf"(active $[{T_REF - WINDOW_L},{T_REF}]$, $L={WINDOW_L}$)"
    ),
    title_s=(
        rf"LagOp LegT at $t={T_REF}$:"
        "\n"
        rf"$\sigma_t(s)$ on $[0,t]$"
        "\n"
        rf"(active $[{T_REF - WINDOW_L},{T_REF}]$)"
    ),
    title_roundtrip=rf"LagOp LegT at $t={T_REF}$: $\sigma_t^{{-1}}(\sigma_t(s))$ roundtrip error",
    title_measure=(
        rf"LagOp LegT at $t={T_REF}$: $\omega_t(s)=\sigma_t'(s)$ "
        rf"(zero on $[0,{T_REF - WINDOW_L})$, uniform on $[{T_REF - WINDOW_L},{T_REF}]$)"
    ),
    warp_family="lagop-legt",
    t_ref=T_REF,
)
hippo_legt = HiPPO(method="legt", N=N_LEGT, dt=DT, T=WINDOW_L)
lag_legt = make_legt_model(N=N_LEGT, dt=DT, T=WINDOW_L, window_length=WINDOW_L)
plot_hippo_lagop_geometry_comparison(
    check="01_2_sigma_warp_legt",
    hippo_model=hippo_legt,
    lag_model=lag_legt,
    family_label="LegT",
)


# Check 01.3: sigma warp + measure/eval geometry (FouT: HiPPO vs LagOp)
fout_basis = make_fout_basis(window_length=WINDOW_L)
plot_sigma_warp_diagnostics(
    check="01_3_sigma_warp_fout",
    sigma_fn=fout_basis.sigma,
    sigma_deriv_fn=fout_basis.sigma_deriv,
    sigma_inv_fn=fout_basis.sigma_inv,
    sigma_inv_deriv_fn=fout_basis.sigma_inv_deriv,
    title_z=(
        rf"LagOp FouT at $t={T_REF}$:"
        "\n"
        rf"$\sigma_t(s)$ on $[0,t]\to z$"
        "\n"
        rf"(active $[{T_REF - WINDOW_L},{T_REF}]$, $L={WINDOW_L}$)"
    ),
    title_s=(
        rf"LagOp FouT at $t={T_REF}$:"
        "\n"
        rf"$\sigma_t(s)$ on $[0,t]$"
        "\n"
        rf"(active $[{T_REF - WINDOW_L},{T_REF}]$)"
    ),
    title_roundtrip=rf"LagOp FouT at $t={T_REF}$: $\sigma_t^{{-1}}(\sigma_t(s))$ roundtrip error",
    title_measure=(
        rf"LagOp FouT at $t={T_REF}$: $\omega_t(s)=\sigma_t'(s)$ "
        rf"(zero on $[0,{T_REF - WINDOW_L})$, uniform on $[{T_REF - WINDOW_L},{T_REF}]$)"
    ),
    warp_family="lagop-fout",
    t_ref=T_REF,
)
hippo_fout = HiPPO(method="fourier", N=N_FOUT, dt=DT, T=WINDOW_L)
lag_fout = make_fout_model(N=N_FOUT, dt=DT, T=WINDOW_L, window_length=WINDOW_L)
plot_hippo_lagop_geometry_comparison(
    check="01_3_sigma_warp_fout",
    hippo_model=hippo_fout,
    lag_model=lag_fout,
    family_label="FouT",
)

# %%
# Check 02: domain expansion / lag shift of a high-order basis mode
# (paper Fig. 3 style; adapted from demo.run_basis_visualization)
# GT = unshifted mode; forward = HiPPO dA (solid) vs LagOp A_∞ (dashed).
check = "02_domain_expansion"
run_domain_expansion_visualization(
    check=check,
    family_label="LegS",
    hippo_method="legs",
    N=N_DOMAIN_EXPANSION,
    dt=DT,
    T=T_DOMAIN,
    phi_func=phi,
    phi_deriv_func=phi_deriv,
    sigma_func=sigma,
    sigma_deriv_func=sigma_deriv,
    sigma_inv_func=sigma_inv,
    sigma_inv_deriv_func=sigma_inv_deriv,
    mode="scale",
    a_tilt="legs",
    file_prefix="legs",
)
run_domain_expansion_visualization(
    check=check,
    family_label="LegT",
    hippo_method="legt",
    N=N_DOMAIN_EXPANSION,
    dt=DT,
    T=T_DOMAIN,
    phi_func=legt_basis.phi,
    phi_deriv_func=legt_basis.phi_deriv,
    sigma_func=legt_basis.sigma,
    sigma_deriv_func=legt_basis.sigma_deriv,
    sigma_inv_func=legt_basis.sigma_inv,
    sigma_inv_deriv_func=legt_basis.sigma_inv_deriv,
    mode="translate",
    window=WINDOW_L,
    a_tilt="translate",
    file_prefix="legt",
)
run_domain_expansion_visualization(
    check=check,
    family_label="FouT",
    hippo_method="fourier",
    N=N_DOMAIN_EXPANSION,
    dt=DT,
    T=T_DOMAIN,
    phi_func=fout_basis.phi,
    phi_deriv_func=fout_basis.phi_deriv,
    sigma_func=fout_basis.sigma,
    sigma_deriv_func=fout_basis.sigma_deriv,
    sigma_inv_func=fout_basis.sigma_inv,
    sigma_inv_deriv_func=fout_basis.sigma_inv_deriv,
    mode="translate",
    window=WINDOW_L,
    a_tilt="translate",
    file_prefix="fout",
)


# %%
# Check 03: LagOp vs original HiPPO (LegS / LegT / FouT)
# SimpleNamespace-style LegS basis for projection diagnostics in the validator.
class _LegSBasisAdapter:
    phi = staticmethod(phi)
    phi_deriv = staticmethod(phi_deriv)
    sigma = staticmethod(sigma)
    sigma_deriv = staticmethod(sigma_deriv)
    sigma_inv = staticmethod(sigma_inv)
    sigma_inv_deriv = staticmethod(sigma_inv_deriv)


legs_basis = _LegSBasisAdapter()
run_translated_ssm_validation(
    check="03_legs_vs_hippo",
    lag_model=lag_legs,
    hippo_method="legs",
    basis=legs_basis,
    N=N_LEGS,
    dt=DT,
    T=T_LONG,
)
run_translated_ssm_validation(
    check="03_legt_vs_hippo",
    lag_model=lag_legt,
    hippo_method="legt",
    basis=legt_basis,
    N=N_LEGT,
    dt=DT,
    T=WINDOW_L,
)
run_translated_ssm_validation(
    check="03_fout_vs_hippo",
    lag_model=lag_fout,
    hippo_method="fourier",
    basis=fout_basis,
    N=N_FOUT,
    dt=DT,
    T=WINDOW_L,
)

# %%
# Check 04: recurrent reconstruction (freq-preserving vs standard)
check = "04_recurrent_reconstruction"
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
    np.linalg.norm(recon_fourier - recon_standard) / np.linalg.norm(recon_standard)
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
    **plot_io(check, "standard_basis_psi.png"),
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
    **plot_io(check, "standard_basis_phi.png"),
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
    **plot_io(check, "freq_basis_psi.png"),
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
    **plot_io(check, "freq_basis_phi.png"),
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
    **plot_io(check, "reconstruction_compare.png"),
)

maybe_save_metrics(
    check,
    "metrics.json",
    {
        "mse_fourier": mse_fourier,
        "mse_standard": mse_standard,
        "relative_reconstruction_diff": rel_diff,
        "recon_mse_between_models": float(
            np.mean((recon_standard - recon_fourier) ** 2)
        ),
    },
)

# %%
# Check 05: direct projection-reconstruction (non-recurrent)
check = "05_projection_reconstruction"
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
    **plot_io(check, "projection_reconstruction.png"),
)

maybe_save_metrics(
    check,
    "metrics.json",
    {
        "mse_exponential": result_exp.mse,
        "mse_freq_preserving": result_freq.mse,
        "runtime_s_exponential": result_exp.runtime_s,
        "runtime_s_freq_preserving": result_freq.runtime_s,
    },
)

# %%
# Check 06: orthonormality via adaptive quadrature (per-window intervals)
check = "06_orthonormality_quad"


def test_orthonormality_quad(phi_func, N, Z=(0.0, 1.0), atol=1e-8):
    failures = []
    for n in range(N):
        for m in range(N):

            def integrand(z, n=n, m=m):
                return phi_func(n, z) * phi_func(m, z)

            result, _ = quad(
                integrand, Z[0], Z[1], epsabs=1e-12, epsrel=1e-12, limit=10000
            )
            if n != m and not np.isclose(result, 0.0, atol=atol):
                failures.append({"n": n, "m": m, "inner_product": float(result)})
    return failures


N_FOURIER_TEST = 5
quad_results = []
print("Check 06: orthonormality (adaptive quadrature)")
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

maybe_save_metrics(check, "metrics.json", {"per_window": quad_results})

# %%
# Check 07: orthonormality via Gauss quadrature + heatmaps
check = "07_orthonormality_gauss"
N_FOURIER_TEST = 20
quad_order = 100 * N_FOURIER_TEST
gauss_results = []

print("Check 07: orthonormality (Gauss quadrature)")
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
            **plot_io(check, f"inner_matrix_N{N}.png"),
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
    **plot_io(check, "inner_matrix_full_0_1_N5.png"),
)

maybe_save_metrics(
    check,
    "metrics.json",
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
