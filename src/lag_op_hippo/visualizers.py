from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _finalize_figure(save_path=None, show=True, save=True, dpi=150, tight=True):
    """Close out a figure: optionally save to disk and/or display in a window.

    Parameters
    ----------
    save_path :
        Destination path when ``save`` is True. Ignored if ``save`` is False
        or ``save_path`` is None.
    show :
        If True, call ``plt.show()`` (interactive window / notebook inline).
        If False, close the figure without displaying.
    save :
        If True and ``save_path`` is set, write the figure to disk.
    tight :
        If True, call ``plt.tight_layout()`` before save/show. Set False when
        the caller already adjusted margins (e.g. outside legends).
    """
    if tight:
        plt.tight_layout()
    if save and save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # bbox_inches='tight' can clip an outside legend; keep full figure canvas.
        plt.savefig(path, dpi=dpi, bbox_inches="tight" if tight else None)
    if show:
        plt.show()
    else:
        plt.close()


def get_plot_config(paper_ready=False):
    """
    Returns a dictionary of plotting settings.

    Args:
        paper_ready (bool, optional): If True, returns settings for publication-quality figures.
                                      Defaults to False.

    Returns:
        dict: A dictionary containing plotting configurations.
    """
    if paper_ready:
        # Settings for a publication-quality figure ✒️
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        plt.rcParams["mathtext.fontset"] = "cm"
        return {
            "axis_label_fontsize": 24,
            "tick_label_fontsize": 20,
            "legend_fontsize": 20,
            "line_width": 2.5,
            "marker_size": 8,
        }
    else:
        # Default settings for exploratory plotting
        plt.rcParams["font.family"] = "sans-serif"
        return {
            "axis_label_fontsize": None,  # Matplotlib default
            "tick_label_fontsize": None,
            "legend_fontsize": None,
            "line_width": 1.5,
            "marker_size": 6,
        }


def plot_reconstructed_signal(
    x,
    original,
    reconstructed,
    data_name="",
    function_name="u",
    domain_name="s",
    title="",
    paper_ready=False,
    recon_labels_list=None,
    recon_linestyle_list=None,
    recon_color_list=None,
    legend_loc="best",
    legend_ncols=None,
    legend_bbox_to_anchor=None,
    legend_outside=False,
    legend_fontsize=None,
    figsize=None,
    xlim=None,
    ylim=None,
    save_path=None,
    show=True,
    save=True,
):
    """
    Plots an original and a reconstructed signal using a shared configuration.

    Args:
        x (array-like): The x-coordinates of the data points.
        original (array-like): The y-coordinates of the original signal.
        reconstructed (array-like): The y-coordinates of the reconstructed signal.
        data_name (str): The name of the data being plotted.
        domain_name (str, optional): The name of the x-axis domain. Defaults to "s".
        title (str, optional): An optional prefix for the plot title. Defaults to "".
        paper_ready (bool, optional): If True, applies publication-quality formatting. Defaults to False.
        legend_outside: False, True/``"below"``, ``"left"``, or ``"right"``
            (legend panel beside the axes so curves are not covered).
        legend_ncols: Number of legend columns.
        legend_bbox_to_anchor: Optional explicit legend anchor.
        legend_fontsize: Optional override for legend text size.
        figsize: Optional ``(w, h)`` figure size in inches.
        save_path: File path used when ``save`` is True.
        show: If True, display the figure in a window / notebook cell.
        save: If True and ``save_path`` is set, write the figure to disk.
    """
    # --- Get the configuration dictionary ---
    config = get_plot_config(paper_ready)

    # Normalize outside placement mode
    if legend_outside is True:
        legend_outside = "below"
    outside_mode = legend_outside if legend_outside else None

    if figsize is None:
        if outside_mode in ("left", "right"):
            figsize = (12.5, 5.0)  # wide: side legend + full-width curve
        elif outside_mode == "below":
            figsize = (9.0, 5.8)
        else:
            figsize = (9.0, 5.0)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    def _series_label(base: str) -> str:
        base = base.strip()
        extra = (data_name or "").strip()
        if not extra:
            return base
        return f"{base} {extra}".strip()

    ax.plot(
        x,
        original,
        label=_series_label("Original"),
        linewidth=config["line_width"],
        color="black",
    )
    if type(reconstructed) == list:
        if recon_labels_list is None:
            recon_labels_list = ["Reconstructed"] * len(reconstructed)
        if recon_linestyle_list is None:
            recon_linestyle_list = ["--"] * len(reconstructed)
        if recon_color_list is None:
            recon_color_list = [None] * len(reconstructed)

        for i, rec in enumerate(reconstructed):
            ax.plot(
                x,
                rec,
                label=_series_label(recon_labels_list[i]),
                linestyle=recon_linestyle_list[i],
                linewidth=config["line_width"],
                color=recon_color_list[i],
            )
    else:
        ax.plot(
            x,
            np.real(reconstructed),
            label=_series_label("Reconstructed"),
            linestyle="--",
            color="red",
            linewidth=config["line_width"],
        )

    # --- Apply text size, labels, and conditional title ---
    ax.set_xlabel(f"{domain_name}", fontsize=config["axis_label_fontsize"])
    ylabel_prefix = (data_name or "").strip()
    ax.set_ylabel(
        f"{ylabel_prefix} {function_name}({domain_name})".strip(),
        fontsize=config["axis_label_fontsize"],
    )
    ax.tick_params(axis="both", labelsize=config["tick_label_fontsize"])

    if paper_ready and title:
        ax.set_title(title, fontsize=config["axis_label_fontsize"])
    elif not paper_ready:
        full_title = (
            title + " " if title else ""
        ) + f"Reconstruction of {data_name} in {domain_name} domain"
        ax.set_title(full_title, fontsize=config["axis_label_fontsize"])
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.grid(True, linestyle="--", alpha=0.6)

    # --- Legend placement ---
    if legend_ncols is None:
        legend_ncols = 1 if outside_mode in ("left", "right") else 2

    if legend_bbox_to_anchor is None and outside_mode == "left":
        # Anchor legend's upper-right corner to the upper-left of the axes.
        legend_loc = "upper right"
        legend_bbox_to_anchor = (-0.04, 1.0)
    elif legend_bbox_to_anchor is None and outside_mode == "right":
        # Anchor legend's upper-left corner to the upper-right of the axes.
        legend_loc = "upper left"
        legend_bbox_to_anchor = (1.02, 1.0)
    elif legend_bbox_to_anchor is None and outside_mode == "below":
        legend_loc = "upper center"
        legend_bbox_to_anchor = (0.5, -0.16)
    elif legend_loc == "best" and legend_bbox_to_anchor is None:
        legend_loc = "upper left"
        legend_bbox_to_anchor = None

    leg_fs = (
        legend_fontsize if legend_fontsize is not None else config["legend_fontsize"]
    )
    # Slightly smaller than axis labels so multi-column legends fit cleanly.
    if leg_fs is None:
        leg_fs = 11
    elif paper_ready and legend_fontsize is None and outside_mode:
        leg_fs = min(int(leg_fs), 12)

    ax.legend(
        fontsize=leg_fs,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncols,
        framealpha=0.95,
        borderaxespad=0.4,
        handlelength=2.2,
        columnspacing=1.0,
    )

    if outside_mode == "left":
        fig.subplots_adjust(left=0.34, right=0.98, top=0.90, bottom=0.12)
        _finalize_figure(save_path=save_path, show=show, save=save, tight=False)
    elif outside_mode == "right":
        fig.subplots_adjust(left=0.10, right=0.72, top=0.90, bottom=0.12)
        _finalize_figure(save_path=save_path, show=show, save=save, tight=False)
    elif outside_mode == "below":
        fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.30)
        _finalize_figure(save_path=save_path, show=show, save=save, tight=False)
    else:
        _finalize_figure(save_path=save_path, show=show, save=save, tight=True)


def plot_basis_functions(
    x,
    Bases,
    ks,
    data_name,
    domain_name="s",
    basis_type="fourier",
    num_functions_front=None,
    num_functions_back=None,
    title="",
    paper_ready=False,
    labels_list=None,
    legend_loc="best",
    save_path=None,
    show=True,
    save=True,
):
    """
    Plot a subset of basis functions against the domain.

    Args:
        x (array-like): Domain values (s or z).
        Bases (np.ndarray): Basis matrix (columns correspond to basis functions).
        ks (array-like): Fourier indices or polynomial degrees.
        data_name (str): Name of the dataset (e.g., "Whitenoise").
        domain_name (str, optional): "s" or "z". Defaults to "s".
        basis_type (str, optional): "fourier", "polynomial", or "legendre". Defaults to "fourier".
        num_functions_front (int, optional): Number of basis functions to plot from the start. Defaults to None.
        num_functions_back (int, optional): Number of basis functions to plot from the end. Defaults to None.
        title (str, optional): An optional prefix for the plot title. Defaults to "".
        paper_ready (bool, optional): If True, applies publication-quality formatting. Defaults to False.
    """
    # --- Get the configuration dictionary ---
    config = get_plot_config(paper_ready)

    plt.figure(figsize=(12, 8))
    N = Bases.shape[1]

    if num_functions_front is None and num_functions_back is None:
        selected_indices = np.arange(N)
    elif num_functions_front is None:
        selected_indices = np.arange(N - num_functions_back, N)
    elif num_functions_back is None:
        selected_indices = np.arange(num_functions_front)
    else:
        selected_indices = np.concatenate(
            (np.arange(num_functions_front), np.arange(N - num_functions_back, N))
        )

    for i in selected_indices:
        if labels_list is not None:
            label = labels_list[i]
        else:
            k = ks[i]
            if basis_type == "legendre":
                label = rf"$P_{{{k}}}({domain_name})$"
            else:
                label = rf"$\phi_{{{k}}}({domain_name})$"
        plt.plot(x, Bases[:, i], label=label, linewidth=config["line_width"])

    # --- Apply text size, labels, and conditional title ---
    plt.xlabel(rf"${domain_name}$", fontsize=config["axis_label_fontsize"])
    plt.ylabel("Basis Function Value", fontsize=config["axis_label_fontsize"])
    plt.xticks(fontsize=config["tick_label_fontsize"])
    plt.yticks(fontsize=config["tick_label_fontsize"])

    if paper_ready and title:
        plt.title(title, fontsize=config["axis_label_fontsize"])
    elif not paper_ready:
        full_title = (
            (title + " " if title else "")
            + f"{data_name} Basis Functions in {domain_name} Domain ({basis_type.capitalize()})"
        )
        plt.title(full_title, fontsize=config["axis_label_fontsize"])

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(
        fontsize=config["legend_fontsize"],
        loc=legend_loc,
    )
    _finalize_figure(save_path=save_path, show=show, save=save)


def _wrap_subplot_title(title: str, max_chars: int = 36) -> str:
    """Insert newlines so long subplot titles do not collide across panels.

    Prefers existing newlines, then breaks before parentheticals / after colons,
    and finally falls back to space-aware wrapping (math segments kept intact
    when possible by not splitting inside ``$...$``).
    """
    if not title:
        return title
    title = str(title).strip()
    if "\n" in title:
        return title

    # Prefer a break before a trailing parenthetical note.
    if " (" in title:
        head, tail = title.split(" (", 1)
        if len(head) >= 12:
            return head + "\n(" + tail

    # Prefer a break after the first colon (e.g. "LagOp FouT at $t=...$: ...").
    if ": " in title:
        head, tail = title.split(": ", 1)
        if len(head) >= 8 and len(tail) >= 8:
            # Further wrap the tail if still long.
            if len(tail) > max_chars and " (" in tail:
                t0, t1 = tail.split(" (", 1)
                tail = t0 + "\n(" + t1
            return head + ":\n" + tail

    # Generic wrap that avoids cutting math mode `$...$`.
    words = title.split(" ")
    lines: list[str] = []
    cur = ""
    in_math = False
    for w in words:
        in_math = in_math ^ (w.count("$") % 2 == 1) if "$" in w else in_math
        trial = w if not cur else cur + " " + w
        if len(trial) <= max_chars or not cur or in_math:
            cur = trial
        else:
            lines.append(cur)
            cur = w
            in_math = w.count("$") % 2 == 1
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def plot_sigma_warp_pair(
    s_vals,
    z_vals,
    sigma_vals,
    sigma_deriv_vals,
    title_s="",
    title_z="",
    labels=None,
    paper_ready=False,
    legend_loc="best",
    save_path=None,
    show=True,
    save=True,
):
    """Side-by-side view of the warp map vs $s$ and vs $z$.

    Left: $\\sigma_t(s)$ and $\\sigma_t'(s)$ against physical time $s$.
    Right: the same series plotted against $z=\\sigma_t(s)$ (canonical domain).
    """
    config = get_plot_config(paper_ready)
    s_vals = np.asarray(s_vals, dtype=np.float64)
    z_vals = np.asarray(z_vals, dtype=np.float64)
    series = [
        np.asarray(sigma_deriv_vals, dtype=np.float64),
        np.asarray(sigma_vals, dtype=np.float64),
    ]
    if labels is None:
        labels = [r"$\sigma_t'(s)$", r"$\sigma_t(s)$"]

    # Slightly shorter titles + extra top margin so multi-line titles do not collide.
    title_fs = config["axis_label_fontsize"]
    if title_fs is not None and title_fs > 14:
        title_fs = max(12, title_fs - 6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    title_s = _wrap_subplot_title(title_s or r"Warp on $s$")
    title_z = _wrap_subplot_title(title_z or r"Warp on $z$")

    # Left: vs s
    for y, lab in zip(series, labels):
        axes[0].plot(s_vals, y, label=lab, linewidth=config["line_width"])
    axes[0].set_xlabel(r"$s$", fontsize=config["axis_label_fontsize"])
    axes[0].set_ylabel("value", fontsize=config["axis_label_fontsize"])
    axes[0].tick_params(labelsize=config["tick_label_fontsize"])
    axes[0].set_title(title_s, fontsize=title_fs, pad=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=config["legend_fontsize"], loc=legend_loc)

    # Right: vs z
    for y, lab in zip(series, labels):
        axes[1].plot(z_vals, y, label=lab, linewidth=config["line_width"])
    axes[1].set_xlabel(r"$z=\sigma_t(s)$", fontsize=config["axis_label_fontsize"])
    axes[1].set_ylabel("value", fontsize=config["axis_label_fontsize"])
    axes[1].tick_params(labelsize=config["tick_label_fontsize"])
    axes[1].set_title(title_z, fontsize=title_fs, pad=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=config["legend_fontsize"], loc=legend_loc)

    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.0)
    _finalize_figure(save_path=save_path, show=show, save=save, tight=False)


def plot_measure_vs_lookback(
    lookback,
    measures,
    labels=None,
    linestyles=None,
    colors=None,
    title="",
    ylabel=r"$\omega(\tau)$",
    xlabel=r"lookback $\tau$ (present at $0$)",
    paper_ready=False,
    legend_loc="best",
    xlim=None,
    ylim=None,
    save_path=None,
    show=True,
    save=True,
):
    """Plot one or more measure densities against lookback time.

    This is the coordinate used by the original HiPPO implementation
    (``vals`` / ``measure``): ``tau = 0`` is the present and larger
    ``tau`` is older history. Multiple series can be overlaid for
    HiPPO vs LagOp comparison.
    """
    config = get_plot_config(paper_ready)
    lookback = np.asarray(lookback, dtype=np.float64)

    if not isinstance(measures, (list, tuple)):
        measures = [measures]
    measures = [np.asarray(m, dtype=np.float64) for m in measures]

    if labels is None:
        labels = [rf"measure[{i}]" for i in range(len(measures))]
    if linestyles is None:
        linestyles = ["-"] * len(measures)
        if len(measures) > 1:
            linestyles = ["-", "-."] + ["--"] * (len(measures) - 2)
    if colors is None:
        colors = [None] * len(measures)
        if len(measures) >= 1:
            colors[0] = "black"
        if len(measures) >= 2:
            colors[1] = "tab:red"

    plt.figure(figsize=(8, 5))
    for measure, label, ls, color in zip(measures, labels, linestyles, colors):
        plt.plot(
            lookback,
            measure,
            label=label,
            linestyle=ls,
            linewidth=config["line_width"],
            color=color,
        )

    plt.xlabel(xlabel, fontsize=config["axis_label_fontsize"])
    plt.ylabel(ylabel, fontsize=config["axis_label_fontsize"])
    plt.xticks(fontsize=config["tick_label_fontsize"])
    plt.yticks(fontsize=config["tick_label_fontsize"])
    if paper_ready and title:
        plt.title(title, fontsize=config["axis_label_fontsize"])
    elif not paper_ready:
        plt.title(
            title or "Measure vs lookback", fontsize=config["axis_label_fontsize"]
        )
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=config["legend_fontsize"], loc=legend_loc)
    _finalize_figure(save_path=save_path, show=show, save=save)


def plot_eval_matrix_columns(
    lookback,
    eval_matrix,
    title="",
    domain_name="s",
    basis_type="eval",
    num_functions_front=None,
    num_functions_back=None,
    paper_ready=False,
    legend_loc="best",
    save_path=None,
    show=True,
    save=True,
    z_vals=None,
    title_s=None,
    title_tau=None,  # alias kept for older call sites
    title_z=None,
    t_end=None,
):
    """Plot eval-matrix columns vs absolute time $s$, optionally also vs $z$.

    Rows of ``eval_matrix`` are indexed by lookback $\\tau$ (present at $\\tau=0$).
    The left panel uses absolute time $s = t_{\\mathrm{end}}-\\tau$ so the present
    sits on the right. When ``z_vals`` is given, the right panel plots the same
    columns vs $z=\\sigma(0,-\\tau)$.
    """
    # Accept either title_s or legacy title_tau
    if title_s is None and title_tau is not None:
        title_s = title_tau

    eval_matrix = np.asarray(eval_matrix, dtype=np.float64)
    lookback = np.asarray(lookback, dtype=np.float64)
    if eval_matrix.ndim != 2:
        raise ValueError(
            f"eval_matrix must be 2D (L, N); got shape {eval_matrix.shape}"
        )
    n_basis = eval_matrix.shape[1]
    n_times = eval_matrix.shape[0]
    if lookback.shape[0] != n_times:
        raise ValueError(
            f"lookback length {lookback.shape[0]} != eval_matrix rows {n_times}"
        )

    # Absolute time: s = t_end - τ  (present τ=0 → s=t_end on the right).
    if t_end is None:
        if n_times >= 2:
            dt = float(lookback[1] - lookback[0])
        else:
            dt = 0.0
        t_end = float(lookback[-1] + dt) if n_times else 0.0
    s_vals = np.asarray(t_end, dtype=np.float64) - lookback

    if num_functions_front is None and num_functions_back is None:
        selected = np.arange(n_basis)
    elif num_functions_front is None:
        selected = np.arange(n_basis - num_functions_back, n_basis)
    elif num_functions_back is None:
        selected = np.arange(min(num_functions_front, n_basis))
    else:
        selected = np.unique(
            np.concatenate(
                (
                    np.arange(min(num_functions_front, n_basis)),
                    np.arange(max(n_basis - num_functions_back, 0), n_basis),
                )
            )
        )

    if z_vals is None:
        return plot_basis_functions(
            x=s_vals,
            Bases=eval_matrix,
            ks=range(n_basis),
            data_name="eval",
            domain_name=domain_name,
            basis_type=basis_type,
            num_functions_front=num_functions_front,
            num_functions_back=num_functions_back,
            title=title,
            paper_ready=paper_ready,
            legend_loc=legend_loc,
            save_path=save_path,
            show=show,
            save=save,
        )

    z_vals = np.asarray(z_vals, dtype=np.float64)
    if z_vals.shape[0] != n_times:
        raise ValueError(
            f"z_vals length {z_vals.shape[0]} != eval_matrix rows {n_times}"
        )

    config = get_plot_config(paper_ready)
    title_fs = config["axis_label_fontsize"]
    if title_fs is not None and title_fs > 14:
        title_fs = max(12, title_fs - 6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    left_title = title_s or title or r"Eval basis vs $s$"
    right_title = title_z or r"Eval basis vs $z=\sigma(0,-(t-s))$"
    if title and not title_s:
        left_title = title + r" (vs $s$)"
    if title and not title_z:
        right_title = title + r" (vs $z$)"

    for i in selected:
        lab = rf"$\phi_{{{i}}}$"
        axes[0].plot(
            s_vals,
            eval_matrix[:, i],
            label=lab,
            linewidth=config["line_width"],
        )
        axes[1].plot(
            z_vals,
            eval_matrix[:, i],
            label=lab,
            linewidth=config["line_width"],
        )

    axes[0].set_xlabel(
        rf"$s$ (present at $t={t_end:g}$)",
        fontsize=config["axis_label_fontsize"],
    )
    axes[0].set_ylabel("basis value", fontsize=config["axis_label_fontsize"])
    axes[0].set_title(left_title, fontsize=title_fs, pad=8)
    axes[0].tick_params(labelsize=config["tick_label_fontsize"])
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=config["legend_fontsize"], loc=legend_loc, ncol=1)

    axes[1].set_xlabel(
        r"$z=\sigma(0,-(t-s))$",
        fontsize=config["axis_label_fontsize"],
    )
    axes[1].set_ylabel("basis value", fontsize=config["axis_label_fontsize"])
    axes[1].set_title(right_title, fontsize=title_fs, pad=8)
    axes[1].tick_params(labelsize=config["tick_label_fontsize"])
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=config["legend_fontsize"], loc=legend_loc, ncol=1)

    fig.tight_layout(w_pad=2.0)
    _finalize_figure(save_path=save_path, show=show, save=save, tight=False)


def lookback_measure_to_absolute_s(lookback, measure, t_ref, s_vals):
    """Map a lookback-domain measure ``ω(τ)`` to absolute time ``s`` via ``τ=t-s``.

    Values with lookback outside the provided grid are set to 0 (correct for
    truncated windows; for LegS the grid usually covers the full horizon).
    """
    lookback = np.asarray(lookback, dtype=np.float64)
    measure = np.asarray(measure, dtype=np.float64)
    s_vals = np.asarray(s_vals, dtype=np.float64)
    tau = np.asarray(t_ref, dtype=np.float64) - s_vals
    out = np.zeros_like(s_vals, dtype=np.float64)
    lo, hi = float(lookback[0]), float(lookback[-1])
    mask = (tau >= lo - 1e-15) & (tau <= hi + 1e-15)
    if np.any(mask):
        out[mask] = np.interp(tau[mask], lookback, measure)
    return out


def plot_measure_density_warp(
    sigma_fn,
    sigma_deriv_fn,
    sigma_inv_fn,
    sigma_inv_deriv_fn,
    t_ref,
    dt,
    title="",
    paper_ready=False,
    legend_loc="best",
    z_ylim=(-0.05, 1.15),
    save_path=None,
    show=True,
    save=True,
    verbose=False,
    lag_label=r"LagOp $\omega_t(s)=\sigma_t'(s)$",
    compare_on_s=None,
):
    """Plot measure density on $s$ and its pushforward on $z$.

    On the time domain the LagOp measure is $\\omega_t(s)=\\sigma_t'(s)=dz/ds$.

    Under the warp $z=\\sigma_t(s)$, change of variables gives
    $\\int u(s)\\,\\omega_t(s)\\,ds = \\int u(\\sigma_t^{-1}(z))\\,dz$, so the
    pushed-forward density
    $\\omega_z(z)=\\sigma_t'(s(z))\\,|ds/dz|$
    must be exactly uniform ($=1$) for any valid warp.

    Parameters
    ----------
    compare_on_s :
        Optional list of ``(label, omega_values)`` arrays aligned to the same
        absolute-time grid as LagOp's ``measure_s`` (length ``len(s_vals)``).
        Typically used to overlay HiPPO's measure after mapping lookback
        ``τ=t-s`` onto absolute ``s`` — absorbs a separate lookback overlay.
    lag_label :
        Legend label for the LagOp curve on the left panel.
    """
    from lag_op_hippo.freq_diagnostics import compute_warp_measure_densities

    result = compute_warp_measure_densities(
        sigma_fn=sigma_fn,
        sigma_deriv_fn=sigma_deriv_fn,
        sigma_inv_fn=sigma_inv_fn,
        sigma_inv_deriv_fn=sigma_inv_deriv_fn,
        t_ref=t_ref,
        dt=dt,
        verbose=verbose,
    )

    config = get_plot_config(paper_ready)
    uniform_ref = np.ones_like(result.z_vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))

    # Left: time-domain measure (LagOp ± optional HiPPO comparison)
    if compare_on_s:
        palette = [("black", "-"), ("tab:blue", "--"), ("tab:green", ":")]
        for i, (lab, vals) in enumerate(compare_on_s):
            vals = np.asarray(vals, dtype=np.float64)
            if vals.shape != result.measure_s.shape:
                raise ValueError(
                    f"compare_on_s[{i}] length {vals.shape} != "
                    f"LagOp grid {result.measure_s.shape}"
                )
            color, ls = palette[i % len(palette)]
            axes[0].plot(
                result.s_vals,
                vals,
                label=lab,
                linewidth=config["line_width"],
                color=color,
                linestyle=ls,
            )
        axes[0].plot(
            result.s_vals,
            result.measure_s,
            label=lag_label,
            linewidth=config["line_width"],
            color="tab:red",
            linestyle="-.",
        )
        left_title = r"Measure on $s$ (HiPPO vs LagOp)"
    else:
        axes[0].plot(
            result.s_vals,
            result.measure_s,
            label=lag_label,
            linewidth=config["line_width"],
            color="black",
            linestyle="-",
        )
        left_title = r"Measure on $s$"

    axes[0].set_xlim(float(result.s_vals[0]), float(result.s_vals[-1]))
    axes[0].set_xlabel(r"$s$", fontsize=config["axis_label_fontsize"])
    axes[0].set_ylabel(r"$\omega_t(s)$", fontsize=config["axis_label_fontsize"])
    axes[0].set_title(left_title, fontsize=config["axis_label_fontsize"])
    axes[0].tick_params(labelsize=config["tick_label_fontsize"])
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=config["legend_fontsize"], loc=legend_loc)

    axes[1].plot(
        result.z_vals,
        result.measure_z,
        label=r"$\omega_z(z)$ on active window",
        linewidth=config["line_width"],
        color="black",
    )
    axes[1].plot(
        result.z_vals,
        uniform_ref,
        label=r"uniform $=1$ (expected)",
        linewidth=config["line_width"],
        linestyle="--",
        color="tab:red",
    )
    if len(result.z_vals) > 0:
        axes[1].set_xlim(float(np.min(result.z_vals)), float(np.max(result.z_vals)))
    axes[1].set_xlabel(r"$z=\sigma_t(s)$", fontsize=config["axis_label_fontsize"])
    axes[1].set_ylabel(r"$\omega_z(z)$", fontsize=config["axis_label_fontsize"])
    # Multi-line title: definition of pushforward density (not just ω(σ^{-1}(z))).
    title_fs = config["axis_label_fontsize"]
    if title_fs is not None and title_fs > 14:
        title_fs = max(11, title_fs - 8)
    axes[1].set_title(
        r"Pushforward $\omega_z(z)=\omega_t(\sigma_t^{-1}(z))\,|ds/dz|$"
        "\n"
        r"(should be uniform $=1$)",
        fontsize=title_fs,
        pad=8,
    )
    axes[1].set_ylim(z_ylim)
    axes[1].tick_params(labelsize=config["tick_label_fontsize"])
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=config["legend_fontsize"], loc=legend_loc)

    # Modest top room for 2-line right-panel title (+ optional suptitle).
    if title:
        fig.suptitle(title, fontsize=config["axis_label_fontsize"], y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.0)
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.0)
    _finalize_figure(save_path=save_path, show=show, save=save, tight=False)
    return {
        "max_pushforward_deviation_from_1": result.max_pushforward_deviation_from_1,
        "mean_pushforward_deviation_from_1": result.mean_pushforward_deviation_from_1,
        "z_min": result.z_min,
        "z_max": result.z_max,
        "is_uniform": result.is_uniform,
    }


def plot_roundtrip_error(
    x,
    error,
    s_roundtrip=None,
    domain_name="s",
    title=r"$|\,s - \sigma_t^{-1}(\sigma_t(s))\,|$",
    ylabel=r"$|\,s - \sigma_t^{-1}(\sigma_t(s))\,|$",
    label=r"roundtrip error",
    paper_ready=False,
    legend_loc="best",
    save_path=None,
    show=True,
    save=True,
):
    """Plot warp roundtrip error, optionally with original and reconstructed $s$.

    Parameters
    ----------
    x :
        Original domain samples $s$ (horizontal axis).
    error :
        Absolute roundtrip error $|s - \\hat s|$.
    s_roundtrip :
        Optional reconstructed samples $\\hat s = \\sigma_t^{-1}(\\sigma_t(s))$.
        When provided, two side-by-side subplots are drawn:
        - left: $s$ (identity) and $\\hat s$;
        - right: absolute error.
        This makes truncation non-invertibility easy to see (old $s$ map to
        the window edge while error grows into the past).
    """
    config = get_plot_config(paper_ready)
    x = np.asarray(x, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)

    if s_roundtrip is None:
        plt.figure(figsize=(8, 5))
        plt.plot(
            x,
            error,
            label=label,
            linewidth=config["line_width"],
            color="black",
        )
        plt.xlabel(rf"${domain_name}$", fontsize=config["axis_label_fontsize"])
        plt.ylabel(ylabel, fontsize=config["axis_label_fontsize"])
        plt.xticks(fontsize=config["tick_label_fontsize"])
        plt.yticks(fontsize=config["tick_label_fontsize"])
        if paper_ready and title:
            plt.title(title, fontsize=config["axis_label_fontsize"])
        elif not paper_ready:
            plt.title(
                title or "Roundtrip error", fontsize=config["axis_label_fontsize"]
            )
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=config["legend_fontsize"], loc=legend_loc)
        _finalize_figure(save_path=save_path, show=show, save=save)
        return

    s_hat = np.asarray(s_roundtrip, dtype=np.float64)
    fig, (ax_s, ax_err) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: original s vs round-tripped ŝ
    ax_s.plot(
        x,
        x,
        label=rf"${domain_name}$ (original)",
        linewidth=config["line_width"],
        color="black",
        linestyle="-",
    )
    ax_s.plot(
        x,
        s_hat,
        label=rf"$\hat{{{domain_name}}}=\sigma_t^{{-1}}(\sigma_t({domain_name}))$",
        linewidth=config["line_width"],
        color="tab:blue",
        linestyle="--",
    )
    ax_s.set_xlabel(rf"${domain_name}$", fontsize=config["axis_label_fontsize"])
    ax_s.set_ylabel(
        rf"${domain_name}$, $\hat{{{domain_name}}}$",
        fontsize=config["axis_label_fontsize"],
    )
    ax_s.tick_params(labelsize=config["tick_label_fontsize"])
    ax_s.set_title(
        rf"Roundtrip value: ${domain_name}$ vs $\hat{{{domain_name}}}$",
        fontsize=config["axis_label_fontsize"],
    )
    ax_s.grid(True, linestyle="--", alpha=0.6)
    ax_s.legend(
        fontsize=config["legend_fontsize"],
        loc=legend_loc if legend_loc != "best" else "upper left",
    )

    # Right: absolute error
    ax_err.plot(
        x,
        error,
        label=label,
        linewidth=config["line_width"],
        color="tab:red",
    )
    ax_err.set_xlabel(rf"${domain_name}$", fontsize=config["axis_label_fontsize"])
    ax_err.set_ylabel(ylabel, fontsize=config["axis_label_fontsize"])
    ax_err.tick_params(labelsize=config["tick_label_fontsize"])
    ax_err.set_title(
        r"Absolute roundtrip error",
        fontsize=config["axis_label_fontsize"],
    )
    ax_err.grid(True, linestyle="--", alpha=0.6)
    ax_err.legend(
        fontsize=config["legend_fontsize"],
        loc=legend_loc if legend_loc != "best" else "upper right",
    )

    if title:
        fig.suptitle(title, fontsize=config["axis_label_fontsize"], y=1.02)

    _finalize_figure(save_path=save_path, show=show, save=save)


def plot_inner_product_matrix(
    inner_mat,
    title="Inner-product matrix",
    paper_ready=False,
    save_path=None,
    show=True,
    save=True,
    vmin=None,
    vmax=None,
):
    """Heatmap of basis inner-product matrix for orthonormality diagnostics."""
    config = get_plot_config(paper_ready)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(inner_mat, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("basis index m", fontsize=config["axis_label_fontsize"])
    plt.ylabel("basis index n", fontsize=config["axis_label_fontsize"])
    if title:
        plt.title(title, fontsize=config["axis_label_fontsize"])
    _finalize_figure(save_path=save_path, show=show, save=save)
