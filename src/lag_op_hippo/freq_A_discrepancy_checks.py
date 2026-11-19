"""Empirical diagnostics for LagOp vs Gu free-map / continuous-A discrepancy.

Checks (numbered in freq_experiment_dev after recurrence):
  05  A_∞ → candidate dA map catalog vs Gu dA (+ optional one-hot free step)
  06  Δt sweep of A_∞ asymptotics and map error
  07  Finite-difference oracle A_emp from re-anchored φ(σ_{t+Δ})
  08  Continuous-A tilt ablations vs Gu A / dA
  09  Eigen / structure of continuous A and discrete dA
"""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
from scipy.linalg import expm, logm

from lag_op_hippo.hippo_by_original_authors import HiPPO
from lag_op_hippo.lag_hippo import (
    LagHiPPO,
    compute_A_infty,
    get_gauss_quadrature,
    vectorized_basis,
)
from lag_op_hippo.visualizers import _finalize_figure, get_plot_config


def _rel_fro(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = np.linalg.norm(b, ord="fro")
    if den < 1e-15:
        return float(np.linalg.norm(a - b, ord="fro"))
    return float(np.linalg.norm(a - b, ord="fro") / den)


def _safe_pinv(m: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(np.asarray(m, dtype=np.float64))


def compute_A_gen_for_basis(
    N: int,
    phi_func: Callable,
    phi_deriv_func: Callable,
    sigma_inv_deriv_func: Callable,
    quad_order: int | None = None,
) -> np.ndarray:
    """A_gen using the *provided* σ_inv' (not the module-level LegS default)."""
    if quad_order is None:
        quad_order = max(2 * N, 4 * N if N > 20 else 2 * N)
    n_vals = np.arange(N)
    z_nodes, z_weights = get_gauss_quadrature(quad_order, 0, 1)
    phi_vals = vectorized_basis(phi_func, n_vals, z_nodes)
    phi_deriv_vals = vectorized_basis(phi_deriv_func, n_vals, z_nodes)
    lag_deriv_factor = 1.0 / np.asarray(
        sigma_inv_deriv_func(0.0, z_nodes), dtype=np.float64
    )
    return np.einsum(
        "i,ik,il->kl", z_weights, phi_vals, lag_deriv_factor[:, None] * phi_deriv_vals
    )


def tilt_continuous_A(A_gen: np.ndarray, a_tilt: str) -> np.ndarray:
    A_gen = np.asarray(A_gen, dtype=np.float64)
    N = A_gen.shape[0]
    if a_tilt == "legs":
        return -(A_gen + np.eye(N)).T
    if a_tilt == "translate":
        return -A_gen
    if a_tilt == "raw":
        return A_gen.copy()
    raise ValueError(f"unknown a_tilt={a_tilt!r}")


def cont2discrete_A(A: np.ndarray, dt: float, method: str = "bilinear") -> np.ndarray:
    N = A.shape[0]
    B = np.zeros((N, 1))
    C = np.ones((1, N))
    D = np.zeros((1,))
    dA, _, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=method)
    return np.asarray(dA, dtype=np.float64)


def build_A_infty_map_catalog(A_infty: np.ndarray, dt: float) -> dict[str, np.ndarray]:
    """Named discrete maps derived from A_∞ (function-side lag Gram)."""
    A_inf = np.asarray(A_infty, dtype=np.float64)
    inv = _safe_pinv(A_inf)
    inv_T = _safe_pinv(A_inf.T)
    e_m = float(np.exp(-dt))
    e_p = float(np.exp(dt))
    return {
        "A_infty": A_inf,
        "A_infty_T": A_inf.T,
        "pinv(A_infty)": inv,
        "pinv(A_infty.T)": inv_T,
        "pinv(A_infty.T)*e^{-dt}": inv_T * e_m,  # LegS-style forward (tilt on)
        "pinv(A_infty)*e^{-dt}": inv * e_m,
        "e^{dt}*A_infty": e_p * A_inf,  # LegS-style backward (tilt on)
        "e^{-dt}*A_infty": e_m * A_inf,
        "e^{dt}*A_infty_T": e_p * A_inf.T,
    }


def one_hot_free_step_rel_error(
    map_mat: np.ndarray,
    dA_ref: np.ndarray,
    mode_idx: int,
) -> float:
    """||M e_n - dA e_n|| / ||dA e_n|| for free coefficient step."""
    N = map_mat.shape[0]
    e = np.zeros(N, dtype=np.float64)
    e[mode_idx] = 1.0
    v_ref = dA_ref @ e
    v = map_mat @ e
    den = np.linalg.norm(v_ref)
    if den < 1e-15:
        return float(np.linalg.norm(v - v_ref))
    return float(np.linalg.norm(v - v_ref) / den)


# ---------------------------------------------------------------------------
# Check 03.01: map catalog
# ---------------------------------------------------------------------------
def run_A_map_catalog_check(
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
    sigma_inv_func,
    sigma_inv_deriv_func,
    a_tilt_default: str,
    mode: str = "scale",
    window: float = 1.0,
    paper_ready: bool = True,
    plot_io_fn,
    save_metrics_fn,
    mode_idx: int | None = None,
):
    """Compare A_∞-derived discrete maps to Gu HiPPO dA (and cont2discrete)."""
    if mode_idx is None:
        mode_idx = max(N - 2, 0)

    hippo = HiPPO(method=hippo_method, N=N, dt=dt, T=T)
    dA_hippo = np.asarray(hippo.dA.detach().cpu().numpy(), dtype=np.float64)
    A_hippo = np.asarray(hippo.A, dtype=np.float64)

    A_infty = compute_A_infty(
        N, phi_func, sigma_func, sigma_inv_func, dt=dt, verbose=False
    )
    A_gen = compute_A_gen_for_basis(
        N, phi_func, phi_deriv_func, sigma_inv_deriv_func
    )
    A_tilt = tilt_continuous_A(A_gen, a_tilt_default)
    dA_from_gen = cont2discrete_A(A_tilt, dt)

    catalog = build_A_infty_map_catalog(A_infty, dt)
    catalog[f"cont2discrete(A_{a_tilt_default})"] = dA_from_gen
    catalog["expm(A_tilt*dt)"] = expm(A_tilt * dt)

    rows = []
    for name, M in catalog.items():
        rows.append(
            {
                "map": name,
                "rel_fro_vs_hippo_dA": _rel_fro(M, dA_hippo),
                "one_hot_rel_err": one_hot_free_step_rel_error(M, dA_hippo, mode_idx),
                "spectral_radius": float(
                    np.max(np.abs(np.linalg.eigvals(M)))
                ),
            }
        )
    rows.sort(key=lambda r: r["rel_fro_vs_hippo_dA"])

    print(f"{check} A_∞ map catalog ({family_label} vs HiPPO-{hippo_method})")
    print(f"  continuous A rel fro vs Gu: {_rel_fro(A_tilt, A_hippo):.6e}")
    for r in rows:
        print(
            f"  {r['map']:28s}  fro={r['rel_fro_vs_hippo_dA']:.4e}  "
            f"e_n={r['one_hot_rel_err']:.4e}  ρ={r['spectral_radius']:.4f}"
        )

    # Bar chart of Frobenius errors
    config = get_plot_config(paper_ready)
    names = [r["map"] for r in rows]
    fro = [r["rel_fro_vs_hippo_dA"] for r in rows]
    en = [r["one_hot_rel_err"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    y = np.arange(len(names))
    axes[0].barh(y, fro, color="tab:blue", alpha=0.85)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlabel(r"$\|M-\mathrm{d}A_{\mathrm{Gu}}\|_F/\|\mathrm{d}A_{\mathrm{Gu}}\|_F$")
    axes[0].set_title(rf"{family_label}: map catalog vs Gu $\mathrm{{d}}A$")
    axes[0].set_xscale("log")
    axes[0].grid(True, axis="x", alpha=0.4)

    axes[1].barh(y, en, color="tab:orange", alpha=0.85)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].set_xlabel(rf"$\|M e_{{{mode_idx}}}-\mathrm{{d}}A e_{{{mode_idx}}}\|/\|\cdot\|$")
    axes[1].set_title(rf"one-hot free step $n={mode_idx}$")
    axes[1].set_xscale("log")
    axes[1].grid(True, axis="x", alpha=0.4)

    fig.suptitle(
        rf"Check 03.01: $A_\infty\mapsto M$ vs Gu $\mathrm{{d}}A$ "
        rf"({family_label}, $N={N}$, $\Delta t={dt}$)",
        fontsize=config["axis_label_fontsize"] or 12,
        y=1.02,
    )
    fig.tight_layout()
    io = plot_io_fn(check, "map_catalog_vs_hippo_dA.png")
    _finalize_figure(
        save_path=io["save_path"], show=io["show"], save=io["save"], tight=False
    )

    metrics = {
        "family": family_label,
        "hippo_method": hippo_method,
        "N": N,
        "dt": dt,
        "T": T,
        "mode_idx": mode_idx,
        "a_tilt_default": a_tilt_default,
        "continuous_A_rel_fro_vs_gu": _rel_fro(A_tilt, A_hippo),
        "dA_from_gen_rel_fro_vs_gu": _rel_fro(dA_from_gen, dA_hippo),
        "catalog": rows,
        "best_map": rows[0]["map"] if rows else None,
        "best_rel_fro": rows[0]["rel_fro_vs_hippo_dA"] if rows else None,
    }
    save_metrics_fn(check, "metrics.json", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Check 03.02: Δt sweep
# ---------------------------------------------------------------------------
def run_dt_sweep_check(
    *,
    check: str,
    family_label: str,
    hippo_method: str,
    N: int,
    T: float,
    phi_func,
    phi_deriv_func,
    sigma_func,
    sigma_inv_func,
    sigma_inv_deriv_func,
    a_tilt: str,
    dts: list[float] | None = None,
    paper_ready: bool = True,
    plot_io_fn,
    save_metrics_fn,
    mode_idx: int | None = None,
):
    if dts is None:
        dts = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    if mode_idx is None:
        mode_idx = max(N - 2, 0)

    A_gen = compute_A_gen_for_basis(
        N, phi_func, phi_deriv_func, sigma_inv_deriv_func
    )
    A_tilt = tilt_continuous_A(A_gen, a_tilt)

    series = {
        "dt": [],
        "rel_Ainf_minus_I_over_dt_vs_Atilt": [],
        "rel_fwd_vs_hippo_dA": [],
        "rel_c2d_vs_hippo_dA": [],
        "one_hot_fwd_vs_hippo": [],
        "rho_fwd": [],
        "rho_hippo_dA": [],
    }

    for dt in dts:
        hippo = HiPPO(method=hippo_method, N=N, dt=dt, T=T)
        dA_h = np.asarray(hippo.dA.detach().cpu().numpy(), dtype=np.float64)
        A_inf = compute_A_infty(
            N, phi_func, sigma_func, sigma_inv_func, dt=dt, verbose=False
        )
        fwd = _safe_pinv(A_inf.T) * np.exp(-dt)
        c2d = cont2discrete_A(A_tilt, dt)
        # Asymptotic generator proxy: (A_∞ - I)/dt
        gen_proxy = (A_inf - np.eye(N)) / dt

        series["dt"].append(float(dt))
        series["rel_Ainf_minus_I_over_dt_vs_Atilt"].append(
            _rel_fro(gen_proxy, A_tilt)
        )
        series["rel_fwd_vs_hippo_dA"].append(_rel_fro(fwd, dA_h))
        series["rel_c2d_vs_hippo_dA"].append(_rel_fro(c2d, dA_h))
        series["one_hot_fwd_vs_hippo"].append(
            one_hot_free_step_rel_error(fwd, dA_h, mode_idx)
        )
        series["rho_fwd"].append(float(np.max(np.abs(np.linalg.eigvals(fwd)))))
        series["rho_hippo_dA"].append(
            float(np.max(np.abs(np.linalg.eigvals(dA_h))))
        )

    print(f"{check} Δt sweep ({family_label})")
    for i, dt in enumerate(series["dt"]):
        print(
            f"  dt={dt:.1e}:  (A∞-I)/dt vs A_tilt fro={series['rel_Ainf_minus_I_over_dt_vs_Atilt'][i]:.3e}  "
            f"fwd vs dA={series['rel_fwd_vs_hippo_dA'][i]:.3e}  "
            f"c2d vs dA={series['rel_c2d_vs_hippo_dA'][i]:.3e}"
        )

    config = get_plot_config(paper_ready)
    dts_a = np.asarray(series["dt"])
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].loglog(
        dts_a,
        series["rel_Ainf_minus_I_over_dt_vs_Atilt"],
        "o-",
        label=r"$(A_\infty-I)/\Delta t$ vs $A_{\mathrm{tilt}}$",
    )
    axes[0].loglog(
        dts_a,
        series["rel_fwd_vs_hippo_dA"],
        "s-",
        label=r"$A_{\mathrm{fwd}}$ vs Gu $\mathrm{d}A$",
    )
    axes[0].loglog(
        dts_a,
        series["rel_c2d_vs_hippo_dA"],
        "^-",
        label=r"$\mathrm{c2d}(A_{\mathrm{tilt}})$ vs Gu $\mathrm{d}A$",
    )
    axes[0].set_xlabel(r"$\Delta t$")
    axes[0].set_ylabel("relative Frobenius error")
    axes[0].set_title(rf"{family_label}: asymptotics / map error")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.4)

    axes[1].semilogx(dts_a, series["rho_fwd"], "o-", label=r"$\rho(A_{\mathrm{fwd}})$")
    axes[1].semilogx(
        dts_a, series["rho_hippo_dA"], "s-", label=r"$\rho(\mathrm{d}A_{\mathrm{Gu}})$"
    )
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel(r"$\Delta t$")
    axes[1].set_ylabel("spectral radius")
    axes[1].set_title(rf"{family_label}: $\rho(\mathrm{{d}}A)$")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.4)

    fig.suptitle(
        rf"Check 03.02: $\Delta t$ sweep ({family_label}, $N={N}$)",
        fontsize=config["axis_label_fontsize"] or 12,
        y=1.02,
    )
    fig.tight_layout()
    io = plot_io_fn(check, "dt_sweep.png")
    _finalize_figure(
        save_path=io["save_path"], show=io["show"], save=io["save"], tight=False
    )

    metrics = {
        "family": family_label,
        "hippo_method": hippo_method,
        "N": N,
        "T": T,
        "a_tilt": a_tilt,
        "mode_idx": mode_idx,
        **series,
    }
    save_metrics_fn(check, "metrics.json", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Check 03.03: FD oracle A_emp
# ---------------------------------------------------------------------------
def run_fd_oracle_A_check(
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
    a_tilt: str,
    mode: str = "scale",
    window: float = 1.0,
    paper_ready: bool = True,
    plot_io_fn,
    save_metrics_fn,
):
    """Estimate A_emp from re-anchored bases φ(σ_t) → φ(σ_{t+Δ}) via eval LS."""
    # LagOp models for eval matrices at T and T+dt
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
    lag_tp = LagHiPPO(
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
    E_t = np.asarray(lag_t.eval_matrix.detach().cpu().numpy(), dtype=np.float64)
    E_tp = np.asarray(lag_tp.eval_matrix.detach().cpu().numpy(), dtype=np.float64)
    # Align lengths: E_tp has one extra lookback sample; drop oldest for match
    # Eval is lookback-ordered (present first). Truncate E_tp to len(E_t).
    L = E_t.shape[0]
    E_tp_aligned = E_tp[:L, :]

    # For each mode n: physical waveform at t is E_t[:, n] (lookback).
    # At t+Δ the re-anchored mode in lookback coords of the *new* present is
    # E_tp[:, n] (first L rows ≈ same τ grid). Coefficients of the old waveform
    # in the new basis: c_plus ≈ lstsq(E_tp_aligned, E_t[:, n]).
    # If free evolution of e_n maps to A_emp e_n ≈ c_plus, then A_emp ≈ [c_plus]_n.
    C_plus = np.zeros((N, N), dtype=np.float64)
    for n in range(N):
        y = E_t[:, n]
        c, _, _, _ = np.linalg.lstsq(E_tp_aligned, y, rcond=None)
        C_plus[:, n] = c

    # A_emp is the discrete free map: e_n ↦ c_plus[:, n], so A_emp = C_plus
    # (columns are images of basis vectors). Continuous proxy: logm(A_emp)/dt
    A_emp_disc = C_plus
    A_gen = compute_A_gen_for_basis(
        N, phi_func, phi_deriv_func, sigma_inv_deriv_func
    )
    A_tilt = tilt_continuous_A(A_gen, a_tilt)
    dA_c2d = cont2discrete_A(A_tilt, dt)
    A_infty = compute_A_infty(
        N, phi_func, sigma_func, sigma_inv_func, dt=dt, verbose=False
    )
    A_fwd = _safe_pinv(A_infty.T) * np.exp(-dt)

    hippo = HiPPO(method=hippo_method, N=N, dt=dt, T=T)
    dA_h = np.asarray(hippo.dA.detach().cpu().numpy(), dtype=np.float64)
    A_h = np.asarray(hippo.A, dtype=np.float64)

    # Continuous FD: logm of discrete map
    try:
        A_emp_cont = np.real(logm(A_emp_disc) / dt)
    except Exception:
        A_emp_cont = np.full_like(A_emp_disc, np.nan)

    comparisons = {
        "A_emp_disc_vs_hippo_dA": _rel_fro(A_emp_disc, dA_h),
        "A_emp_disc_vs_A_fwd": _rel_fro(A_emp_disc, A_fwd),
        "A_emp_disc_vs_c2d": _rel_fro(A_emp_disc, dA_c2d),
        "A_emp_cont_vs_A_tilt": _rel_fro(A_emp_cont, A_tilt)
        if np.isfinite(A_emp_cont).all()
        else float("nan"),
        "A_emp_cont_vs_hippo_A": _rel_fro(A_emp_cont, A_h)
        if np.isfinite(A_emp_cont).all()
        else float("nan"),
        "A_tilt_vs_hippo_A": _rel_fro(A_tilt, A_h),
        "A_fwd_vs_hippo_dA": _rel_fro(A_fwd, dA_h),
    }

    print(f"{check} FD oracle A_emp ({family_label})")
    for k, v in comparisons.items():
        print(f"  {k}: {v:.6e}")

    config = get_plot_config(paper_ready)
    labels = list(comparisons.keys())
    vals = [comparisons[k] for k in labels]
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    y = np.arange(len(labels))
    ax.barh(y, vals, color="tab:purple", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("relative Frobenius error")
    ax.set_title(
        rf"Check 03.03: FD oracle $A_{{\mathrm{{emp}}}}$ ({family_label}, $N={N}$)"
    )
    ax.grid(True, axis="x", alpha=0.4)
    fig.tight_layout()
    io = plot_io_fn(check, "fd_oracle_A_emp.png")
    _finalize_figure(
        save_path=io["save_path"], show=io["show"], save=io["save"], tight=False
    )

    # Heatmap residual A_emp - dA_hippo
    fig2, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    mats = [
        (A_emp_disc, r"$A_{\mathrm{emp}}^{\mathrm{disc}}$"),
        (dA_h, r"Gu $\mathrm{d}A$"),
        (A_emp_disc - dA_h, r"$A_{\mathrm{emp}}-\mathrm{d}A$"),
    ]
    for ax, (M, title) in zip(axes, mats):
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r")
        ax.set_title(title, fontsize=10)
        fig2.colorbar(im, ax=ax, fraction=0.046)
    fig2.suptitle(rf"{family_label}: FD free map vs Gu $\mathrm{{d}}A$", y=1.03)
    fig2.tight_layout()
    io2 = plot_io_fn(check, "fd_oracle_heatmaps.png")
    _finalize_figure(
        save_path=io2["save_path"], show=io2["show"], save=io2["save"], tight=False
    )

    metrics = {
        "family": family_label,
        "hippo_method": hippo_method,
        "N": N,
        "dt": dt,
        "T": T,
        "a_tilt": a_tilt,
        **comparisons,
    }
    save_metrics_fn(check, "metrics.json", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Check 03.04: continuous A tilt ablations
# ---------------------------------------------------------------------------
def run_tilt_ablation_check(
    *,
    check: str,
    family_label: str,
    hippo_method: str,
    N: int,
    dt: float,
    T: float,
    phi_func,
    phi_deriv_func,
    sigma_inv_deriv_func,
    paper_ready: bool = True,
    plot_io_fn,
    save_metrics_fn,
    tilts: tuple[str, ...] = ("raw", "legs", "translate"),
    disc_methods: tuple[str, ...] = ("bilinear", "zoh", "euler"),
):
    hippo = HiPPO(method=hippo_method, N=N, dt=dt, T=T)
    A_h = np.asarray(hippo.A, dtype=np.float64)
    dA_h = np.asarray(hippo.dA.detach().cpu().numpy(), dtype=np.float64)
    A_gen = compute_A_gen_for_basis(
        N, phi_func, phi_deriv_func, sigma_inv_deriv_func
    )

    rows = []
    for tilt in tilts:
        A = tilt_continuous_A(A_gen, tilt)
        row = {
            "a_tilt": tilt,
            "rel_A_vs_gu": _rel_fro(A, A_h),
            "skew_rel": float(
                np.linalg.norm(A + A.T, ord="fro")
                / max(np.linalg.norm(A, ord="fro"), 1e-15)
            ),
        }
        for meth in disc_methods:
            try:
                dA = cont2discrete_A(A, dt, method=meth)
                row[f"rel_dA_{meth}_vs_gu"] = _rel_fro(dA, dA_h)
            except Exception as exc:  # noqa: BLE001
                row[f"rel_dA_{meth}_vs_gu"] = float("nan")
                row[f"err_{meth}"] = str(exc)
        # exact expm
        dA_ex = expm(A * dt)
        row["rel_dA_expm_vs_gu"] = _rel_fro(dA_ex, dA_h)
        rows.append(row)

    print(f"{check} tilt ablation ({family_label})")
    for r in rows:
        print(
            f"  tilt={r['a_tilt']:10s}  A_fro={r['rel_A_vs_gu']:.4e}  "
            f"bilinear_dA={r.get('rel_dA_bilinear_vs_gu', float('nan')):.4e}  "
            f"expm_dA={r['rel_dA_expm_vs_gu']:.4e}  skew={r['skew_rel']:.4e}"
        )

    config = get_plot_config(paper_ready)
    tilt_names = [r["a_tilt"] for r in rows]
    x = np.arange(len(tilt_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    series = [
        ("rel_A_vs_gu", r"$A$ vs Gu $A$"),
        ("rel_dA_bilinear_vs_gu", r"c2d-bilinear"),
        ("rel_dA_zoh_vs_gu", r"c2d-zoh"),
        ("rel_dA_expm_vs_gu", r"$\mathrm{expm}(A\Delta t)$"),
    ]
    for i, (key, lab) in enumerate(series):
        vals = [r.get(key, np.nan) for r in rows]
        ax.bar(x + (i - 1.5) * width, vals, width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(tilt_names)
    ax.set_yscale("log")
    ax.set_ylabel("relative Frobenius error")
    ax.set_title(
        rf"Check 03.04: tilt / discretization ablations ({family_label}, $N={N}$)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    io = plot_io_fn(check, "tilt_ablation.png")
    _finalize_figure(
        save_path=io["save_path"], show=io["show"], save=io["save"], tight=False
    )

    metrics = {
        "family": family_label,
        "hippo_method": hippo_method,
        "N": N,
        "dt": dt,
        "T": T,
        "ablations": rows,
        "best_tilt_by_A": min(rows, key=lambda r: r["rel_A_vs_gu"])["a_tilt"],
        "best_tilt_by_dA_bilinear": min(
            rows, key=lambda r: r.get("rel_dA_bilinear_vs_gu", np.inf)
        )["a_tilt"],
    }
    save_metrics_fn(check, "metrics.json", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Check 03.05: eigen / structure
# ---------------------------------------------------------------------------
def run_eigen_structure_check(
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
    sigma_inv_func,
    sigma_inv_deriv_func,
    a_tilt: str,
    paper_ready: bool = True,
    plot_io_fn,
    save_metrics_fn,
):
    hippo = HiPPO(method=hippo_method, N=N, dt=dt, T=T)
    A_h = np.asarray(hippo.A, dtype=np.float64)
    dA_h = np.asarray(hippo.dA.detach().cpu().numpy(), dtype=np.float64)

    A_gen = compute_A_gen_for_basis(
        N, phi_func, phi_deriv_func, sigma_inv_deriv_func
    )
    A_lag = tilt_continuous_A(A_gen, a_tilt)
    A_inf = compute_A_infty(
        N, phi_func, sigma_func, sigma_inv_func, dt=dt, verbose=False
    )
    A_fwd = _safe_pinv(A_inf.T) * np.exp(-dt)

    def _eigs(M):
        w = np.linalg.eigvals(np.asarray(M, dtype=np.float64))
        return w

    e_lag = _eigs(A_lag)
    e_h = _eigs(A_h)
    e_fwd = _eigs(A_fwd)
    e_dh = _eigs(dA_h)

    # Sort by real part (continuous) / abs (discrete)
    e_lag_s = e_lag[np.argsort(e_lag.real)]
    e_h_s = e_h[np.argsort(e_h.real)]
    e_fwd_s = e_fwd[np.argsort(-np.abs(e_fwd))]
    e_dh_s = e_dh[np.argsort(-np.abs(e_dh))]

    rank_inf = int(np.linalg.matrix_rank(A_inf, tol=1e-8))
    skew_lag = float(
        np.linalg.norm(A_lag + A_lag.T, ord="fro")
        / max(np.linalg.norm(A_lag, ord="fro"), 1e-15)
    )
    skew_h = float(
        np.linalg.norm(A_h + A_h.T, ord="fro")
        / max(np.linalg.norm(A_h, ord="fro"), 1e-15)
    )

    metrics_core = {
        "family": family_label,
        "hippo_method": hippo_method,
        "N": N,
        "dt": dt,
        "a_tilt": a_tilt,
        "skew_rel_A_lag": skew_lag,
        "skew_rel_A_gu": skew_h,
        "rho_A_fwd": float(np.max(np.abs(e_fwd))),
        "rho_dA_gu": float(np.max(np.abs(e_dh))),
        "rank_A_infty": rank_inf,
        "rank_deficit_A_infty": int(N - rank_inf),
        "max_re_eig_A_lag": float(np.max(e_lag.real)),
        "max_re_eig_A_gu": float(np.max(e_h.real)),
        "cond_A_lag": float(np.linalg.cond(A_lag)),
        "cond_A_gu": float(np.linalg.cond(A_h)),
        "eig_A_lag_real": e_lag_s.real.tolist(),
        "eig_A_lag_imag": e_lag_s.imag.tolist(),
        "eig_A_gu_real": e_h_s.real.tolist(),
        "eig_A_gu_imag": e_h_s.imag.tolist(),
    }
    print(f"{check} eigen/structure ({family_label})")
    for k in (
        "skew_rel_A_lag",
        "skew_rel_A_gu",
        "rho_A_fwd",
        "rho_dA_gu",
        "rank_A_infty",
        "rank_deficit_A_infty",
        "max_re_eig_A_lag",
        "max_re_eig_A_gu",
    ):
        print(f"  {k}: {metrics_core[k]}")

    config = get_plot_config(paper_ready)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].scatter(
        e_h_s.real, e_h_s.imag, c="tab:blue", s=28, label="Gu $A$", alpha=0.85
    )
    axes[0].scatter(
        e_lag_s.real,
        e_lag_s.imag,
        c="tab:red",
        s=28,
        marker="x",
        label=rf"LagOp $A$ (tilt={a_tilt})",
        alpha=0.85,
    )
    axes[0].axvline(0, color="gray", lw=0.8)
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set_xlabel(r"$\mathrm{Re}\,\lambda$")
    axes[0].set_ylabel(r"$\mathrm{Im}\,\lambda$")
    axes[0].set_title("continuous $A$ spectrum")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.4)
    axes[0].set_aspect("equal", adjustable="datalim")

    axes[1].scatter(
        e_dh_s.real, e_dh_s.imag, c="tab:blue", s=28, label=r"Gu $\mathrm{d}A$", alpha=0.85
    )
    axes[1].scatter(
        e_fwd_s.real,
        e_fwd_s.imag,
        c="tab:red",
        s=28,
        marker="x",
        label=r"LagOp $A_{\mathrm{fwd}}$",
        alpha=0.85,
    )
    th = np.linspace(0, 2 * np.pi, 200)
    axes[1].plot(np.cos(th), np.sin(th), "k--", lw=0.8, alpha=0.5, label=r"$|\lambda|=1$")
    axes[1].set_xlabel(r"$\mathrm{Re}\,\mu$")
    axes[1].set_ylabel(r"$\mathrm{Im}\,\mu$")
    axes[1].set_title("discrete free-map spectrum")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.4)
    axes[1].set_aspect("equal", adjustable="datalim")

    fig.suptitle(
        rf"Check 03.05: eigen/structure ({family_label}, $N={N}$)",
        fontsize=config["axis_label_fontsize"] or 12,
        y=1.02,
    )
    fig.tight_layout()
    io = plot_io_fn(check, "eigen_structure.png")
    _finalize_figure(
        save_path=io["save_path"], show=io["show"], save=io["save"], tight=False
    )

    save_metrics_fn(check, "metrics.json", metrics_core)
    return metrics_core


def run_all_discrepancy_checks_for_family(
    *,
    family_key: str,
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
    a_tilt: str,
    mode: str,
    window: float,
    paper_ready: bool,
    plot_io_fn,
    save_metrics_fn,
):
    """Run Checks 05–09 for one LagOp family vs Gu HiPPO twin."""
    prefix = family_key  # legs / legt / fout
    out = {}
    out["05"] = run_A_map_catalog_check(
        check=f"03_01_A_map_catalog_{prefix}",
        family_label=family_label,
        hippo_method=hippo_method,
        N=N,
        dt=dt,
        T=T,
        phi_func=phi_func,
        phi_deriv_func=phi_deriv_func,
        sigma_func=sigma_func,
        sigma_inv_func=sigma_inv_func,
        sigma_inv_deriv_func=sigma_inv_deriv_func,
        a_tilt_default=a_tilt,
        mode=mode,
        window=window,
        paper_ready=paper_ready,
        plot_io_fn=plot_io_fn,
        save_metrics_fn=save_metrics_fn,
    )
    out["06"] = run_dt_sweep_check(
        check=f"03_02_dt_sweep_{prefix}",
        family_label=family_label,
        hippo_method=hippo_method,
        N=N,
        T=T,
        phi_func=phi_func,
        phi_deriv_func=phi_deriv_func,
        sigma_func=sigma_func,
        sigma_inv_func=sigma_inv_func,
        sigma_inv_deriv_func=sigma_inv_deriv_func,
        a_tilt=a_tilt,
        paper_ready=paper_ready,
        plot_io_fn=plot_io_fn,
        save_metrics_fn=save_metrics_fn,
    )
    out["07"] = run_fd_oracle_A_check(
        check=f"03_03_fd_oracle_{prefix}",
        family_label=family_label,
        hippo_method=hippo_method,
        N=N,
        dt=dt,
        T=T,
        phi_func=phi_func,
        phi_deriv_func=phi_deriv_func,
        sigma_func=sigma_func,
        sigma_deriv_func=sigma_deriv_func,
        sigma_inv_func=sigma_inv_func,
        sigma_inv_deriv_func=sigma_inv_deriv_func,
        a_tilt=a_tilt,
        mode=mode,
        window=window,
        paper_ready=paper_ready,
        plot_io_fn=plot_io_fn,
        save_metrics_fn=save_metrics_fn,
    )
    out["08"] = run_tilt_ablation_check(
        check=f"03_04_tilt_ablation_{prefix}",
        family_label=family_label,
        hippo_method=hippo_method,
        N=N,
        dt=dt,
        T=T,
        phi_func=phi_func,
        phi_deriv_func=phi_deriv_func,
        sigma_inv_deriv_func=sigma_inv_deriv_func,
        paper_ready=paper_ready,
        plot_io_fn=plot_io_fn,
        save_metrics_fn=save_metrics_fn,
    )
    out["09"] = run_eigen_structure_check(
        check=f"03_05_eigen_structure_{prefix}",
        family_label=family_label,
        hippo_method=hippo_method,
        N=N,
        dt=dt,
        T=T,
        phi_func=phi_func,
        phi_deriv_func=phi_deriv_func,
        sigma_func=sigma_func,
        sigma_inv_func=sigma_inv_func,
        sigma_inv_deriv_func=sigma_inv_deriv_func,
        a_tilt=a_tilt,
        paper_ready=paper_ready,
        plot_io_fn=plot_io_fn,
        save_metrics_fn=save_metrics_fn,
    )
    return out
