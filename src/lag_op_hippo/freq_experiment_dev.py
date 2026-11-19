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
from lag_op_hippo.freq_A_discrepancy_checks import (
    run_all_discrepancy_checks_for_family,
)
import matplotlib.pyplot as plt

from lag_op_hippo.visualizers import (
    _finalize_figure,
    get_plot_config,
    lookback_measure_to_absolute_s,
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
T_RECON = 3.0  # horizon for Check 02.01/02.02 reconstruction (u and û on s∈[0,T_RECON])
T_DOMAIN = 3.0  # horizon for domain-expansion (paper Fig. 3 style)
DOMAIN_ZOOM = 10  # samples near the present for lag-shift zoom
PAPER_READY_PLOTS = True

# =============================================================================
# I/O toggles — edit these (env vars override the defaults if set)
# =============================================================================
# SHOW_PLOTS: open / display each figure in a window (or notebook cell).
# SAVE_PLOTS: write PNGs under outputs/freq_dev/latest/<check>/.
# SAVE_METRICS: write metrics.json / run_config.json alongside figures.
# ARCHIVE_RUN: copy latest/ → runs/<timestamp>/ after the full script finishes.
_env_bool = lambda key, default: (os.environ.get(key, "1" if default else "0") != "0")
SHOW_PLOTS = _env_bool("SHOW_PLOTS", False)
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
    sigma_func=None,
):
    """Plot measure vs lookback and/or eval-matrix columns for one model.

    Eval matrices are shown side-by-side vs lookback $\\tau$ and vs
    $z=\\sigma(0,-\\tau)$ when ``sigma_func`` is provided.
    """
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
        z_vals = None
        if sigma_func is not None:
            # LagOp / HiPPO eval_matrix is indexed by lookback τ; z = σ(0, -τ).
            z_vals = np.asarray(sigma_func(0.0, -lookback), dtype=np.float64)
        plot_eval_matrix_columns(
            lookback,
            eval_matrix,
            title=rf"{model_label}: eval (first {n_show})",
            title_s=rf"{model_label}: vs $s$ (present on the right)",
            title_z=rf"{model_label}: vs $z=\sigma(0,-(t-s))$",
            basis_type=file_prefix,
            num_functions_front=n_show,
            paper_ready=PAPER_READY_PLOTS,
            z_vals=z_vals,
            **plot_io(check, f"{file_prefix}_eval_matrix.png"),
        )


def plot_hippo_lagop_geometry_comparison(
    *,
    check: str,
    hippo_model: HiPPO | None,
    lag_model: LagHiPPO,
    family_label: str,
    t_ref: float | None = None,
    title_measure: str | None = None,
):
    """Eval-matrix visuals + (when HiPPO exists) measure comparison.

    HiPPO vs LagOp measures are absorbed into the left panel of
    ``measure_density.png`` (absolute time ``s``), so a separate
    ``measure_vs_lookback_overlay.png`` is not written.  The right panel of
    that figure remains the LagOp pushforward-to-``z`` diagnostic.

    If there is no HiPPO counterpart (e.g. freq-preserving), LagOp measure is
    already plotted by ``plot_sigma_warp_diagnostics``; here we only plot eval.
    """
    lag_lookback = _as_numpy(lag_model.vals)
    lag_measure = _as_numpy(lag_model.measure)
    lag_eval = _as_numpy(lag_model.eval_matrix)
    t_ref = float(T_REF if t_ref is None else t_ref)

    # Use LagOp warp to map lookback → z for both models (same z-axis for fair panels).
    sigma_for_z = lag_model.sigma_func

    if hippo_model is None:
        plot_model_geometry(
            check=check,
            lookback=lag_lookback,
            measure=lag_measure,
            eval_matrix=lag_eval,
            model_label=f"LagOp {family_label}",
            file_prefix="lagop",
            plot_measure=False,  # measure already in measure_density.png
            plot_eval=True,
            sigma_func=sigma_for_z,
        )
        return

    hippo_lookback = _as_numpy(hippo_model.vals)
    hippo_measure = _as_numpy(hippo_model.measure)
    hippo_eval = _as_numpy(hippo_model.eval_matrix)

    # Per-model eval matrices (no basis overlay): each as (τ | z) side-by-side.
    plot_model_geometry(
        check=check,
        lookback=lag_lookback,
        measure=lag_measure,
        eval_matrix=lag_eval,
        model_label=f"LagOp {family_label}",
        file_prefix="lagop",
        plot_measure=False,
        plot_eval=True,
        sigma_func=sigma_for_z,
    )
    plot_model_geometry(
        check=check,
        lookback=hippo_lookback,
        measure=hippo_measure,
        eval_matrix=hippo_eval,
        model_label=f"HiPPO {family_label}",
        file_prefix="hippo",
        plot_measure=False,
        plot_eval=True,
        sigma_func=sigma_for_z,
    )

    # Rewrite measure_density.png: left = HiPPO vs LagOp on absolute s;
    # right = LagOp pushforward on z (unchanged diagnostic).
    s_grid = np.arange(0.0, t_ref, DT, dtype=np.float64)
    hippo_on_s = lookback_measure_to_absolute_s(
        hippo_lookback, hippo_measure, t_ref, s_grid
    )
    if title_measure is None:
        title_measure = (
            rf"{family_label}: $\omega_t(s)$ on $s$ (HiPPO vs LagOp) "
            rf"+ pushforward on $z$"
        )
    plot_measure_density_warp(
        sigma_fn=lag_model.sigma_func,
        sigma_deriv_fn=lag_model.sigma_deriv_func,
        sigma_inv_fn=lag_model.sigma_inv_func,
        sigma_inv_deriv_fn=lag_model.sigma_inv_deriv_func,
        t_ref=t_ref,
        dt=DT,
        title=title_measure,
        paper_ready=PAPER_READY_PLOTS,
        lag_label=rf"LagOp {family_label}",
        compare_on_s=[(rf"HiPPO {family_label}", hippo_on_s)],
        verbose=False,
        **plot_io(check, "measure_density.png"),
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
    plot_measure_density: bool = True,
):
    """Shared sigma-warp visuals: warp map, roundtrip, and optional measure density.

    Set ``plot_measure_density=False`` when a later HiPPO comparison will
    rewrite ``measure_density.png`` (avoids a LagOp-only intermediate figure
    in Jupyter while still overwriting the same path on disk).
    """
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

    measure_stats = {}
    if plot_measure_density:
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
            "measure_density_plotted_here": plot_measure_density,
            **measure_stats,
        },
    )


def _cosine_test_signal(T: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Shared test input for Check 02.01/02.02: cos so u≈1 near the present."""
    vals = np.arange(0.0, T, dt)
    u = np.cos(2.0 * np.pi * 3.0 * vals)
    return vals, u


def _project_via_eval_matrix(eval_matrix, u_absolute: np.ndarray) -> np.ndarray:
    """Least-squares projection of absolute-time signal onto eval_matrix columns.

    ``eval_matrix`` rows are ordered by lookback τ (present at row 0).  Absolute
    signal ``u_absolute`` is ordered by s (present at the end), so we flip to
    lookback order, solve

        min_c || Eval c - u_lookback ||_2 ,

    and return coefficients ``c`` (length N). Reconstruction is then
    ``reconstruct(c)`` / ``Eval @ c`` (then flip), not a recurrence.
    """
    E = np.asarray(eval_matrix, dtype=np.float64)
    u_lb = np.flip(np.asarray(u_absolute, dtype=np.float64)).copy()
    if E.shape[0] != u_lb.shape[0]:
        raise ValueError(
            f"eval_matrix rows {E.shape[0]} != signal length {u_lb.shape[0]}"
        )
    c, _, _, _ = np.linalg.lstsq(E, u_lb, rcond=None)
    return c


# Legend / metrics names for Check 02.01–02.02 reconstruction plots.
_HIPPO_METHOD_LEGEND = {
    "legs": "HiPPO-LegS",
    "legt": "HiPPO-LegT",
    "fourier": "HiPPO-FouT",
}
_LAG_METHOD_LEGEND = {
    "legs": "LagOp-LegS",
    "legt": "LagOp-LegT",
    "fourier": "LagOp-FouT",
}


def _hippo_legend_label(hippo_method: str) -> str:
    return _HIPPO_METHOD_LEGEND.get(hippo_method, f"HiPPO-{hippo_method}")


def _lag_legend_label(hippo_method: str) -> str:
    return _LAG_METHOD_LEGEND.get(hippo_method, f"LagOp-{hippo_method}")


def run_projection_roundtrip_validation(
    *,
    check: str,
    lag_model: LagHiPPO,
    hippo_method: str,
    basis,
    N: int,
    dt: float,
    T: float,
    lag_label: str | None = None,
    hippo_label: str | None = None,
    title_family: str | None = None,
):
    """Sanity check: Eval-matrix projection roundtrip (no recurrence).

    Math:
      u_lookback = flip(u)
      c = argmin ||Eval c - u_lookback||   (least squares)
      û = reconstruct(c) = flip(Eval c)

    Same plot/metrics style as :func:`run_translated_ssm_validation`, but
    ``c`` is obtained by static projection onto the eval basis rather than by
    scanning ``u`` with ``c ← dA c + dB u``.

    For families without a Gu HiPPO twin (e.g. freq-preserving), pass
    ``hippo_method='legs'`` and set ``title_family`` / labels so the comparison
    is clearly LagOp-FouP vs HiPPO-LegS.
    """
    hippo = HiPPO(method=hippo_method, N=N, dt=dt, T=T)
    vals, u = _cosine_test_signal(T, dt)
    family = title_family if title_family is not None else hippo_method
    if hippo_label is None:
        hippo_label = _hippo_legend_label(hippo_method)
    if lag_label is None:
        lag_label = _lag_legend_label(hippo_method)

    c_hippo = _project_via_eval_matrix(hippo.eval_matrix, u)
    c_lag = _project_via_eval_matrix(lag_model.eval_matrix, u)

    dtype_h = hippo.eval_matrix.dtype
    dtype_l = lag_model.eval_matrix.dtype
    recon_hippo = hippo.reconstruct(torch.as_tensor(c_hippo, dtype=dtype_h)).numpy()
    recon_lag = lag_model.reconstruct(torch.as_tensor(c_lag, dtype=dtype_l)).numpy()

    # Also report continuous LagOp projection on Z (Gauss), for cross-check.
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
        "mode": "eval_matrix_projection_roundtrip",
        "hippo_method": hippo_method,
        "lag_family": family,
        "lag_label": lag_label,
        "hippo_label": hippo_label,
        "N": N,
        "dt": dt,
        "T": T,
        "mse_hippo_vs_input": float(np.mean((recon_hippo - u) ** 2)),
        "mse_lag_vs_input": float(np.mean((recon_lag - u) ** 2)),
        "mse_hippo_vs_lag": float(np.mean((recon_hippo - recon_lag) ** 2)),
        "rel_error_hippo_vs_lag": float(
            np.linalg.norm(recon_hippo - recon_lag)
            / max(np.linalg.norm(recon_hippo), 1e-12)
        ),
        "gauss_projection_recon_mse": proj.mse,
        "orthogonality_max_offdiag": ortho["max_offdiag"],
        "orthogonality_max_diag_deviation": ortho["max_diag_deviation"],
    }
    print(f"{check} projection roundtrip ({family})")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    plot_reconstructed_signal(
        vals,
        u,
        [recon_hippo, recon_lag],
        data_name="",
        domain_name="s",
        # Discrete samples of ψ on the lookback grid (s-domain), not ∫ φ U dz on Z.
        title=(
            rf"Projection roundtrip on $s$ via $\psi$ ({family}, $N={N}$)"
            "\n"
            r"$\hat{u}=\sum_{n=0}^{N-1}\langle\psi_n,\,u\rangle_{\omega_t}\,\psi_n$"
        ),
        paper_ready=PAPER_READY_PLOTS,
        figsize=(11.0, 6.8),
        recon_labels_list=[hippo_label, lag_label],
        recon_linestyle_list=["--", "-."],
        function_name="u",
        legend_loc="upper left",
        **plot_io(check, "reconstruction_vs_hippo.png"),
    )
    maybe_save_metrics(check, "metrics.json", metrics)


def _lag_recurrence_ab_mathtex(lag_model: LagHiPPO) -> tuple[str, str]:
    """Mathtext symbols for the discrete maps actually used by ``lag_model``.

    - ``derive_A_from='gen'`` → ``dA_gen`` (cont2discrete of tilted ``A_gen``)
    - ``derive_A_from='infty'`` → ``A_∞`` (lag-op end-to-end discrete map)
    - ``input_hold='delta'`` → ``dB_δ`` (cont2discrete of ``B_delta``)
    - ``input_hold='zoh'`` → ``B_zoh``
    """
    derive = getattr(lag_model, "derive_A_from", "gen")
    hold = getattr(lag_model, "input_hold", "delta")
    if derive == "infty":
        dA_tex = r"A_{\infty}"
    elif derive == "gen":
        dA_tex = r"\mathrm{d}A_{\mathrm{gen}}"
    else:
        dA_tex = rf"\mathrm{{d}}A_{{\mathrm{{{derive}}}}}"
    if hold == "zoh":
        dB_tex = r"B_{\mathrm{zoh}}"
    elif hold == "delta":
        dB_tex = r"\mathrm{d}B_{\delta}"
    else:
        dB_tex = rf"\mathrm{{d}}B_{{\mathrm{{{hold}}}}}"
    return dA_tex, dB_tex


def run_translated_ssm_validation(
    *,
    check: str,
    lag_model: LagHiPPO,
    hippo_method: str,
    basis,
    N: int,
    dt: float,
    T: float,
    lag_label: str | None = None,
    hippo_label: str | None = None,
    title_family: str | None = None,
):
    """Compare LagOp SSM recurrence vs original HiPPO (LegS / LegT / FouT).

    Scans u with c ← dA c + dB u up to c_T, then reconstructs history from c_T.

    For freq-preserving LagOp (no Gu twin), compare against ``hippo_method='legs'``
    and set ``title_family`` / series labels accordingly.
    """
    hippo = HiPPO(method=hippo_method, N=N, dt=dt, T=T)
    vals, u = _cosine_test_signal(T, dt)
    u_torch = torch.tensor(u, dtype=torch.float)
    family = title_family if title_family is not None else hippo_method
    if hippo_label is None:
        hippo_label = _hippo_legend_label(hippo_method)
    if lag_label is None:
        lag_label = _lag_legend_label(hippo_method)
    dA_tex, dB_tex = _lag_recurrence_ab_mathtex(lag_model)

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
        "mode": "recurrent_scan_to_c_T",
        "hippo_method": hippo_method,
        "lag_family": family,
        "lag_label": lag_label,
        "hippo_label": hippo_label,
        "derive_A_from": getattr(lag_model, "derive_A_from", None),
        "input_hold": getattr(lag_model, "input_hold", None),
        "a_tilt": getattr(lag_model, "a_tilt", None),
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
        "mse_hippo_vs_input": float(np.mean((recon_hippo - u) ** 2)),
        "mse_lag_vs_input": float(np.mean((recon_lag - u) ** 2)),
        "gauss_projection_recon_mse": proj.mse,
        "orthogonality_max_offdiag": ortho["max_offdiag"],
        "orthogonality_max_diag_deviation": ortho["max_diag_deviation"],
    }
    print(f"{check} recurrence validation ({family})")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    plot_reconstructed_signal(
        vals,
        u,
        [recon_hippo, recon_lag],
        data_name="",
        domain_name="s",
        title=(
            rf"Recurrence scan $\to c_T\to$ reconstruct ({family}, $N={N}$)"
            "\n"
            rf"$\hat{{u}}=\sum_{{n=0}}^{{N-1}}(c_T)_n\,\psi_n$ "
            rf"($c_{{t+1}} = {dA_tex}\,c_t+{dB_tex}\,u_{{t+1}}$)"
        ),
        paper_ready=PAPER_READY_PLOTS,
        figsize=(11.0, 7.2),
        recon_labels_list=[hippo_label, lag_label],
        # GT is solid black; keep HiPPO dashed so it does not hide the input.
        recon_linestyle_list=["--", "-."],
        function_name="u",
        legend_loc="upper left",
        **plot_io(check, "reconstruction_vs_hippo.png"),
    )
    maybe_save_metrics(check, "metrics.json", metrics)


def _projected_basis_mode_on_s(
    *,
    phi_func,
    sigma_func,
    sigma_deriv_func,
    mode_idx: int,
    s_abs: np.ndarray,
    t_present: float,
    mode: str,
) -> np.ndarray:
    """Evaluate basis mode ``φ_n(σ_{t}(s))`` on absolute time ``s``.

    Matches LagHiPPO eval convention: lookback ``τ = t - s``,
    ``z = σ_0(-τ)``, optional translate-window zeroing via ``σ'``.
    Future samples (``s > t``) are set to NaN.
    """
    s_abs = np.asarray(s_abs, dtype=np.float64)
    tau = float(t_present) - s_abs
    out = np.full(s_abs.shape, np.nan, dtype=np.float64)
    valid = tau >= 0.0
    if not np.any(valid):
        return out
    z = np.asarray(sigma_func(0.0, -tau[valid]), dtype=np.float64)
    out[valid] = np.asarray(phi_func(mode_idx, z), dtype=np.float64)
    if mode == "translate":
        # Outside the active window the basis is truncated (measure zero).
        w = np.asarray(sigma_deriv_func(0.0, -tau), dtype=np.float64)
        out = np.where(w > 0.0, out, 0.0)
        out = np.where(valid, out, np.nan)
    return out


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
    tilt: bool = True,
    example_n: int | None = None,
    zoom: int = DOMAIN_ZOOM,
    file_prefix: str | None = None,
):
    """Paper Fig. 3-style lag / domain-expansion of one high-order basis mode.

    Side-by-side panels (forward | backward).  Each panel shows:

    - unshifted GT at present ``t``: ``φ_n(σ_t(s))`` (LagOp eval column)
    - **projected GT** under the shifted warp: ``φ_n(σ_{t±Δ}(s))``
    - **HiPPO GT (unshifted)**: Gu ``eval_matrix @ e_n`` (same one-hot mode)
    - operator maps: HiPPO ``dA`` / ``dA^{-1}`` and LagOp ``A_∞`` forward /
      backward applied to a one-hot coefficient, then reconstructed

    ``tilt`` (default True) multiplies the LagOp maps by scalar
    ``e^{±Δt}`` factors, independent of family / ``a_tilt``::

        forward  = (A_∞^T)^+  e^{-Δt}   if tilt else (A_∞^T)^+
        backward = e^{Δt} A_∞           if tilt else A_∞

    ``example_n`` is the coefficient / basis index of the one-hot mode (shown
    in the figure title as n=...). Default is N-2.
    """
    prefix = file_prefix or family_label.lower().replace(" ", "_")
    # Coefficient index of the one-hot mode used for the lag visualization.
    # Default N-2 matches demo.py (high-order mode; for FouT avoids the null sin(0) slot at n=1).
    if example_n is None:
        mode_idx = max(N - 2, 0)
    else:
        mode_idx = int(example_n)
    if not (0 <= mode_idx < N):
        raise ValueError(f"basis index mode_idx={mode_idx} out of range for N={N}")

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
    # Optional scalar e^{±Δt} tilt (default on); not tied to model family / a_tilt.
    A_infty_T_inv = np.linalg.pinv(A_infty.T)
    if tilt:
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

    # Present anchors match discrete grids: lag_t ends at T-dt, lag_t_plus at T.
    t_now = T - dt
    t_fwd = T
    t_bwd = T - 2.0 * dt

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
    s_abs = vals_t_plus.detach().cpu().numpy().astype(np.float64)
    zoom_n = min(zoom, len(vals_t_plus))
    sl = slice(-zoom_n, None)

    # Projected GTs: true re-anchored bases φ(σ_{t±Δ}(s)), not operator maps.
    gt_proj_fwd = _projected_basis_mode_on_s(
        phi_func=phi_func,
        sigma_func=sigma_func,
        sigma_deriv_func=sigma_deriv_func,
        mode_idx=mode_idx,
        s_abs=s_abs,
        t_present=t_fwd,
        mode=mode,
    )
    gt_proj_bwd = _projected_basis_mode_on_s(
        phi_func=phi_func,
        sigma_func=sigma_func,
        sigma_deriv_func=sigma_deriv_func,
        mode_idx=mode_idx,
        s_abs=s_abs,
        t_present=t_bwd,
        mode=mode,
    )
    # Cross-check: projected GT at t_now should match lag GT on the shared support.
    gt_proj_now = _projected_basis_mode_on_s(
        phi_func=phi_func,
        sigma_func=sigma_func,
        sigma_deriv_func=sigma_deriv_func,
        mode_idx=mode_idx,
        s_abs=s_abs,
        t_present=t_now,
        mode=mode,
    )

    # --- Side-by-side: forward | backward (fewer overlays per panel) ---
    config = get_plot_config(PAPER_READY_PLOTS)
    lw = config["line_width"]
    title_fs = config["axis_label_fontsize"]
    if title_fs is not None and title_fs > 14:
        title_fs = max(12, int(title_fs) - 6)
    leg_fs = config["legend_fontsize"]
    if leg_fs is not None and leg_fs > 12:
        leg_fs = 11

    x = np.asarray(vals_t_plus[sl].detach().cpu().numpy(), dtype=np.float64)
    y_gt = np.asarray(gt_lag[sl].detach().cpu().numpy(), dtype=np.float64)
    # HiPPO GT = Gu unshifted mode column: flip(pad(eval_matrix @ e_n)).
    y_gt_hippo = np.asarray(gt_hippo[sl].detach().cpu().numpy(), dtype=np.float64)
    y_fwd_hippo = np.asarray(fwd_hippo[sl].detach().cpu().numpy(), dtype=np.float64)
    y_fwd_lag = np.asarray(fwd_lag[sl].detach().cpu().numpy(), dtype=np.float64)
    y_bwd_hippo = np.asarray(bwd_hippo[sl].detach().cpu().numpy(), dtype=np.float64)
    y_bwd_lag = np.asarray(bwd_lag[sl].detach().cpu().numpy(), dtype=np.float64)
    y_proj_fwd = gt_proj_fwd[sl]
    y_proj_bwd = gt_proj_bwd[sl]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True)
    ax_fwd, ax_bwd = axes

    # GTs in the background; operator forward/backward maps on top.
    # Styles: GT φ solid black; projected φ(σ_{t±Δ}) black solid + markers;
    # HiPPO GT (eval_matrix @ e_n) blue dashed.
    gt_kw = dict(color="black", linestyle="-", linewidth=lw, zorder=1, alpha=0.9)
    proj_kw = dict(
        color="black",
        linestyle="-",
        linewidth=lw,
        marker="o",
        markersize=8.0,
        markevery=1,
        zorder=1,
        alpha=0.9,
    )
    hippo_gt_kw = dict(
        color="tab:blue",
        linestyle="--",
        linewidth=lw,
        zorder=1,
        alpha=0.9,
    )
    op_kw = dict(linewidth=lw, zorder=3)
    # HiPPO operator maps: same style in both panels (blue marker-dash).
    hippo_op_kw = dict(
        color="tab:blue",
        linestyle="-.",
        linewidth=lw,
        marker="o",
        markersize=8.0,
        markevery=1,
        zorder=3,
    )
    if tilt:
        lag_fwd_label = r"LagOp $A_\infty\,e^{-\Delta t}$ forward"
        lag_bwd_label = r"LagOp $e^{\Delta t}A_\infty$ backward"
    else:
        lag_fwd_label = r"LagOp $A_\infty$ forward"
        lag_bwd_label = r"LagOp $A_\infty$ backward"

    def _mark_time_interval(ax, t_a: float, t_b: float, label_a: str, label_b: str):
        """Gray band + vertical lines for the lag step ``[t_a, t_b]`` (order-independent)."""
        lo, hi = (t_a, t_b) if t_a <= t_b else (t_b, t_a)
        ax.axvspan(lo, hi, facecolor="0.85", edgecolor="none", zorder=0, alpha=0.7)
        ax.axvline(
            t_a, color="0.35", linestyle="-", linewidth=1.2, zorder=2, alpha=0.85
        )
        ax.axvline(
            t_b, color="0.35", linestyle="-", linewidth=1.2, zorder=2, alpha=0.85
        )
        # Light labels at the top of the axes (in axes coordinates for y).
        ann_kw = dict(
            xycoords=("data", "axes fraction"),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=max(9, (leg_fs or 11) - 1),
            color="0.25",
            zorder=4,
        )
        ax.annotate(label_a, xy=(t_a, 1.0), xytext=(0, 2), **ann_kw)
        ax.annotate(label_b, xy=(t_b, 1.0), xytext=(0, 2), **ann_kw)

    # Forward panel: GTs first (back), then operator projections (front).
    _mark_time_interval(ax_fwd, t_now, t_fwd, r"$t$", r"$t+\Delta$")
    ax_fwd.plot(x, y_gt, **gt_kw, label=r"GT $\phi(\sigma_t)$")
    ax_fwd.plot(
        x,
        y_proj_fwd,
        **proj_kw,
        label=r"GT $\phi(\sigma_{t+\Delta})$",
    )
    ax_fwd.plot(
        x,
        y_gt_hippo,
        **hippo_gt_kw,
        label=r"GT HiPPO (unshifted)",
    )
    ax_fwd.plot(
        x,
        y_fwd_hippo,
        **hippo_op_kw,
        label=r"HiPPO $dA$ forward",
    )
    ax_fwd.plot(
        x,
        y_fwd_lag,
        color="tab:red",
        linestyle="--",
        **op_kw,
        label=lag_fwd_label,
    )
    ax_fwd.set_title(r"Forward lag ($t\to t+\Delta$)", fontsize=title_fs, pad=4)
    ax_fwd.set_xlabel(r"$s$", fontsize=config["axis_label_fontsize"])
    ax_fwd.set_ylabel(
        rf"$\phi_{{{mode_idx}}}(s)$", fontsize=config["axis_label_fontsize"]
    )

    # Backward panel: same draw order.
    _mark_time_interval(ax_bwd, t_now, t_bwd, r"$t$", r"$t-\Delta$")
    ax_bwd.plot(x, y_gt, **gt_kw, label=r"GT $\phi(\sigma_t)$")
    ax_bwd.plot(
        x,
        y_proj_bwd,
        **proj_kw,
        label=r"GT $\phi(\sigma_{t-\Delta})$",
    )
    ax_bwd.plot(
        x,
        y_gt_hippo,
        **hippo_gt_kw,
        label=r"GT HiPPO (unshifted)",
    )
    ax_bwd.plot(
        x,
        y_bwd_hippo,
        **hippo_op_kw,
        label=r"HiPPO $dA^{-1}$ backward",
    )
    ax_bwd.plot(
        x,
        y_bwd_lag,
        color="tab:red",
        linestyle="--",
        **op_kw,
        label=lag_bwd_label,
    )
    ax_bwd.set_title(r"Backward lag ($t\to t-\Delta$)", fontsize=title_fs, pad=4)
    ax_bwd.set_xlabel(r"$s$", fontsize=config["axis_label_fontsize"])

    for ax in axes:
        ax.tick_params(labelsize=config["tick_label_fontsize"])
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=leg_fs, loc="best", framealpha=0.95)

    # Main title: context on line 1; generic lag map on line 2 (A named in legend).
    # Manual top margin: tight_layout(rect) left a large empty band under the
    # suptitle; subplots_adjust keeps plot titles close to the main title.
    fig.suptitle(
        rf"{family_label}: domain expansion "
        rf"(basis $n={mode_idx}$, $N={N}$; "
        rf"projected GT $\phi(\sigma_{{t\pm\Delta}})$ + HiPPO $dA$ vs LagOp $A_\infty$)"
        "\n"
        r"$\psi_{t\pm\Delta}=A\,\psi_t$",
        fontsize=title_fs,
        y=0.98,
        linespacing=1.2,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.84, wspace=0.22)
    io = plot_io(check, f"{prefix}_domain_expansion.png")
    _finalize_figure(
        save_path=io["save_path"],
        show=io["show"],
        save=io["save"],
        tight=False,
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
        "basis_index": mode_idx,
        "N_basis": N,
        "a_tilt": a_tilt,
        "tilt": bool(tilt),
        "mode": mode,
        "zoom": zoom_n,
        "t_now": t_now,
        "t_fwd": t_fwd,
        "t_bwd": t_bwd,
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
        "rel_err_gt_lag_vs_proj_now": _rel_err(gt_lag, gt_proj_now),
        "rel_err_fwd_lag_vs_proj_fwd": _rel_err(fwd_lag, gt_proj_fwd),
        "rel_err_fwd_hippo_vs_proj_fwd": _rel_err(fwd_hippo, gt_proj_fwd),
        "rel_err_bwd_lag_vs_proj_bwd": _rel_err(bwd_lag, gt_proj_bwd),
        "rel_err_bwd_hippo_vs_proj_bwd": _rel_err(bwd_hippo, gt_proj_bwd),
        "rel_err_gt_hippo_vs_proj_now": _rel_err(gt_hippo, gt_proj_now),
    }
    print(f"{check} domain expansion ({family_label})")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    maybe_save_metrics(check, f"{prefix}_metrics.json", metrics)
    return metrics


# %%
# Check 01.01.0: sigma warp and roundtrip (frequency-preserving)
# No original HiPPO counterpart — LagOp geometry only.
plot_sigma_warp_diagnostics(
    check="01_01_0_sigma_warp",
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
    check="01_01_0_sigma_warp",
    hippo_model=None,
    lag_model=freq_model,
    family_label="Freq-preserving",
)


# Check 01.01.1: sigma warp + measure/eval geometry (LegS: HiPPO vs LagOp)
plot_sigma_warp_diagnostics(
    check="01_01_1_sigma_warp_lagop",
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
    # measure_density.png written later with HiPPO vs LagOp overlay
    plot_measure_density=False,
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
    check="01_01_1_sigma_warp_lagop",
    hippo_model=hippo_legs,
    lag_model=lag_legs,
    family_label="LegS",
)


# Check 01.01.2: sigma warp + measure/eval geometry (LegT: HiPPO vs LagOp)
legt_basis = make_legt_basis(window_length=WINDOW_L)
plot_sigma_warp_diagnostics(
    check="01_01_2_sigma_warp_legt",
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
    plot_measure_density=False,  # HiPPO vs LagOp version written below
)
hippo_legt = HiPPO(method="legt", N=N_LEGT, dt=DT, T=WINDOW_L)
lag_legt = make_legt_model(N=N_LEGT, dt=DT, T=WINDOW_L, window_length=WINDOW_L)
plot_hippo_lagop_geometry_comparison(
    check="01_01_2_sigma_warp_legt",
    hippo_model=hippo_legt,
    lag_model=lag_legt,
    family_label="LegT",
)


# Check 01.01.3: sigma warp + measure/eval geometry (FouT: HiPPO vs LagOp)
fout_basis = make_fout_basis(window_length=WINDOW_L)
plot_sigma_warp_diagnostics(
    check="01_01_3_sigma_warp_fout",
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
    plot_measure_density=False,  # HiPPO vs LagOp version written below
)
hippo_fout = HiPPO(method="fourier", N=N_FOUT, dt=DT, T=WINDOW_L)
lag_fout = make_fout_model(N=N_FOUT, dt=DT, T=WINDOW_L, window_length=WINDOW_L)
plot_hippo_lagop_geometry_comparison(
    check="01_01_3_sigma_warp_fout",
    hippo_model=hippo_fout,
    lag_model=lag_fout,
    family_label="FouT",
)

# %%
# Check 01.02: domain expansion / lag shift of a high-order basis mode
# (paper Fig. 3 style; adapted from demo.run_basis_visualization)
# Matched pairs: LagOp-LegS/LegT/FouT vs Gu HiPPO twin; LagOp-FouP vs HiPPO-FouT
# (FouP = frequency-preserving Fourier; no Gu twin — FouT is the closer baseline).
# tilt=True: LagOp maps × e^{±Δt} (independent of family).
check = "01_02_domain_expansion"
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
    tilt=True,
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
    tilt=True,
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
    tilt=True,
    file_prefix="fout",
)
# FouP (frequency-preserving Fourier) has no Gu twin → compare lag maps / eval vs HiPPO-FouT.
run_domain_expansion_visualization(
    check=check,
    family_label="FouP vs FouT",
    hippo_method="fourier",
    N=N_DOMAIN_EXPANSION,
    dt=DT,
    T=T_DOMAIN,
    phi_func=basis.phi,
    phi_deriv_func=basis.phi_deriv,
    sigma_func=basis.sigma,
    sigma_deriv_func=basis.sigma_deriv,
    sigma_inv_func=basis.sigma_inv,
    sigma_inv_deriv_func=basis.sigma_inv_deriv,
    mode="scale",
    window=L_FOURIER,
    a_tilt="legs",
    tilt=True,
    file_prefix="foup",
)


# %%
# Shared basis adapter + models for Check 02.01 (projection) and Check 02.02 (recurrence).
class _LegSBasisAdapter:
    phi = staticmethod(phi)
    phi_deriv = staticmethod(phi_deriv)
    sigma = staticmethod(sigma)
    sigma_deriv = staticmethod(sigma_deriv)
    sigma_inv = staticmethod(sigma_inv)
    sigma_inv_deriv = staticmethod(sigma_inv_deriv)


legs_basis = _LegSBasisAdapter()

# Shared LagOp models for Check 02.01 (projection) and Check 02.02 (recurrence).
lag_legs_recon = LagHiPPO(
    N=N_LEGS,
    phi_func=phi,
    phi_deriv_func=phi_deriv,
    sigma_func=sigma,
    sigma_deriv_func=sigma_deriv,
    sigma_inv_func=sigma_inv,
    sigma_inv_deriv_func=sigma_inv_deriv,
    dt=DT,
    T=T_RECON,
)
lag_legt_recon = make_legt_model(N=N_LEGT, dt=DT, T=T_RECON, window_length=WINDOW_L)
lag_fout_recon = make_fout_model(N=N_FOUT, dt=DT, T=T_RECON, window_length=WINDOW_L)
# FouP (frequency-preserving Fourier) has no Gu HiPPO twin: compare vs HiPPO-LegS
# at the same N. Use N_LEGS so capacity matches the LegS Check 02.01/02.02 runs.
# derive_A_from='gen' (not A_∞): A_∞ + ZOH is unstable on this cosine scan and
# yields NaN c_T; gen matches the stable LagHiPPO-LegS reconstruction path.
foup_basis_recon = make_frequency_preserving_basis(
    alpha=ALPHA,
    window_length=L_FOURIER,
    eps=EPS,
)
lag_foup_recon = make_frequency_preserving_model(
    N=N_LEGS,
    dt=DT,
    T=T_RECON,
    alpha=ALPHA,
    window_length=L_FOURIER,
    eps=EPS,
    derive_A_from="gen",
    input_hold="zoh",
    quad_order=200 * N_LEGS,
)

# %%
# Check 02.01: Eval-matrix projection roundtrip (no recurrence)
#   c = lstsq(Eval, flip(u)),  û = reconstruct(c)
# Isolates projection stability vs the recurrent scan in Check 02.02.
run_projection_roundtrip_validation(
    check="02_01_legs_proj_roundtrip",
    lag_model=lag_legs_recon,
    hippo_method="legs",
    basis=legs_basis,
    N=N_LEGS,
    dt=DT,
    T=T_RECON,
)
run_projection_roundtrip_validation(
    check="02_01_legt_proj_roundtrip",
    lag_model=lag_legt_recon,
    hippo_method="legt",
    basis=legt_basis,
    N=N_LEGT,
    dt=DT,
    T=T_RECON,
)
run_projection_roundtrip_validation(
    check="02_01_fout_proj_roundtrip",
    lag_model=lag_fout_recon,
    hippo_method="fourier",
    basis=fout_basis,
    N=N_FOUT,
    dt=DT,
    T=T_RECON,
)
run_projection_roundtrip_validation(
    check="02_01_foup_proj_roundtrip",
    lag_model=lag_foup_recon,
    hippo_method="legs",  # no Gu FouP twin; baseline = HiPPO-LegS
    basis=foup_basis_recon,
    N=N_LEGS,
    dt=DT,
    T=T_RECON,
    lag_label="LagOp-FouP",
    hippo_label="HiPPO-LegS",
    title_family="FouP vs legs",
)

# %%
# Check 02.02: recurrence scan → c_T → reconstruct (was Check 03)
run_translated_ssm_validation(
    check="02_02_legs_vs_hippo",
    lag_model=lag_legs_recon,
    hippo_method="legs",
    basis=legs_basis,
    N=N_LEGS,
    dt=DT,
    T=T_RECON,
)
run_translated_ssm_validation(
    check="02_02_legt_vs_hippo",
    lag_model=lag_legt_recon,
    hippo_method="legt",
    basis=legt_basis,
    N=N_LEGT,
    dt=DT,
    T=T_RECON,
)
run_translated_ssm_validation(
    check="02_02_fout_vs_hippo",
    lag_model=lag_fout_recon,
    hippo_method="fourier",
    basis=fout_basis,
    N=N_FOUT,
    dt=DT,
    T=T_RECON,
)
run_translated_ssm_validation(
    check="02_02_foup_vs_hippo",
    lag_model=lag_foup_recon,
    hippo_method="legs",  # no Gu FouP twin; baseline = HiPPO-LegS
    basis=foup_basis_recon,
    N=N_LEGS,
    dt=DT,
    T=T_RECON,
    lag_label="LagOp-FouP",
    hippo_label="HiPPO-LegS",
    title_family="FouP vs legs",
)

# %%
# Checks 05–09: A / free-map discrepancy diagnostics (per Gu-matched family)
#   05 map catalog A_∞ → M vs Gu dA
#   06 Δt sweep asymptotics
#   07 FD oracle A_emp from re-anchored φ(σ_{t+Δ})
#   08 continuous-A tilt + discretization ablations
#   09 eigen / structure of A and dA
# Uses N_LEGS/LEGT/FOUT (same capacity as Check 02.01/02.02 recon).
_N_DIAG = N_LEGS
_T_DIAG = T_RECON
for _fam in (
    dict(
        family_key="legs",
        family_label="LegS",
        hippo_method="legs",
        N=_N_DIAG,
        phi_func=phi,
        phi_deriv_func=phi_deriv,
        sigma_func=sigma,
        sigma_deriv_func=sigma_deriv,
        sigma_inv_func=sigma_inv,
        sigma_inv_deriv_func=sigma_inv_deriv,
        a_tilt="legs",
        mode="scale",
        window=1.0,
    ),
    dict(
        family_key="legt",
        family_label="LegT",
        hippo_method="legt",
        N=N_LEGT,
        phi_func=legt_basis.phi,
        phi_deriv_func=legt_basis.phi_deriv,
        sigma_func=legt_basis.sigma,
        sigma_deriv_func=legt_basis.sigma_deriv,
        sigma_inv_func=legt_basis.sigma_inv,
        sigma_inv_deriv_func=legt_basis.sigma_inv_deriv,
        a_tilt="translate",
        mode="translate",
        window=WINDOW_L,
    ),
    dict(
        family_key="fout",
        family_label="FouT",
        hippo_method="fourier",
        N=N_FOUT,
        phi_func=fout_basis.phi,
        phi_deriv_func=fout_basis.phi_deriv,
        sigma_func=fout_basis.sigma,
        sigma_deriv_func=fout_basis.sigma_deriv,
        sigma_inv_func=fout_basis.sigma_inv,
        sigma_inv_deriv_func=fout_basis.sigma_inv_deriv,
        a_tilt="translate",
        mode="translate",
        window=WINDOW_L,
    ),
):
    run_all_discrepancy_checks_for_family(
        **_fam,
        dt=DT,
        T=_T_DIAG,
        paper_ready=PAPER_READY_PLOTS,
        plot_io_fn=plot_io,
        save_metrics_fn=maybe_save_metrics,
    )

# %%
# Check 02.03: recurrent reconstruction (freq-preserving vs standard)
# (formerly Check 05)
check = "02_03_recurrent_reconstruction"
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
# Check 02.04: direct projection-reconstruction (non-recurrent)
# (formerly Check 06)
check = "02_04_projection_reconstruction"
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
# Check 04.01: orthonormality via adaptive quadrature (per-window intervals)
# (formerly Check 07)
check = "04_01_orthonormality_quad"


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
print("Check 04.01: orthonormality (adaptive quadrature)")
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
# Check 04.02: orthonormality via Gauss quadrature + heatmaps
# (formerly Check 08)
check = "04_02_orthonormality_gauss"
N_FOURIER_TEST = 20
quad_order = 100 * N_FOURIER_TEST
gauss_results = []

print("Check 04.02: orthonormality (Gauss quadrature)")
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
