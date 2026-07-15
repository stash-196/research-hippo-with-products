from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _finalize_figure(save_path=None, show=True, dpi=150):
    plt.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
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
    xlim=None,
    ylim=None,
    save_path=None,
    show=True,
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
    """
    # --- Get the configuration dictionary ---
    config = get_plot_config(paper_ready)

    # --- Create the figure and plot the data ---
    plt.figure(figsize=(8, 5))
    plt.plot(
        x,
        original,
        label=f"Original {data_name}",  # {function_name}({domain_name})",
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
            plt.plot(
                x,
                rec,
                label=f"{recon_labels_list[i]} {data_name}",  # {function_name}({domain_name})",
                linestyle=recon_linestyle_list[i],
                linewidth=config["line_width"],
                color=recon_color_list[i],
            )
    else:
        plt.plot(
            x,
            np.real(reconstructed),
            label=f"Reconstructed {data_name}",  # {function_name}({domain_name})",
            linestyle="--",
            color="red",
            linewidth=config["line_width"],
        )

    # --- Apply text size, labels, and conditional title ---
    plt.xlabel(f"{domain_name}", fontsize=config["axis_label_fontsize"])
    plt.ylabel(
        f"{data_name} {function_name}({domain_name})",
        fontsize=config["axis_label_fontsize"],
    )
    plt.xticks(fontsize=config["tick_label_fontsize"])
    plt.yticks(fontsize=config["tick_label_fontsize"])

    if paper_ready and title:
        plt.title(title, fontsize=config["axis_label_fontsize"])
    elif not paper_ready:
        full_title = (
            title + " " if title else ""
        ) + f"Reconstruction of {data_name} in {domain_name} domain"
        plt.title(full_title, fontsize=config["axis_label_fontsize"])
    # y limits
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(
        fontsize=config["legend_fontsize"],
        loc=legend_loc if legend_loc != "best" else "upper center",
        bbox_to_anchor=(0.5, 1.1) if legend_loc == "best" else None,
        ncol=2,
    )
    _finalize_figure(save_path=save_path, show=show)


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
    _finalize_figure(save_path=save_path, show=show)


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
    verbose=False,
):
    """Plot measure density on $s$ and its pushforward on $z$.

    On the time domain the LagOp measure is $\\omega_t(s)=\\sigma_t'(s)=dz/ds$.

    Under the warp $z=\\sigma_t(s)$, change of variables gives
    $\\int u(s)\\,\\omega_t(s)\\,ds = \\int u(\\sigma_t^{-1}(z))\\,dz$, so the
    pushed-forward density
    $\\omega_z(z)=\\sigma_t'(s(z))\\,|ds/dz|$
    must be exactly uniform ($=1$) for any valid warp.
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        result.s_vals,
        result.measure_s,
        label=r"$\omega_t(s)=\sigma_t'(s)$",
        linewidth=config["line_width"],
        color="black",
    )
    axes[0].set_xlim(float(result.s_vals[0]), float(result.s_vals[-1]))
    axes[0].set_xlabel(r"$s$", fontsize=config["axis_label_fontsize"])
    axes[0].set_ylabel(r"$\omega_t(s)$", fontsize=config["axis_label_fontsize"])
    axes[0].set_title(
        r"Measure on $s$",
        fontsize=config["axis_label_fontsize"],
    )
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
    axes[1].set_title(
        r"Pushed-forward measure on $z$ (should be uniform)",
        fontsize=config["axis_label_fontsize"],
    )
    axes[1].set_ylim(z_ylim)
    axes[1].tick_params(labelsize=config["tick_label_fontsize"])
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=config["legend_fontsize"], loc=legend_loc)

    if title:
        fig.suptitle(title, fontsize=config["axis_label_fontsize"], y=1.02)

    _finalize_figure(save_path=save_path, show=show)
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
    domain_name="s",
    title=r"$|\,s - \sigma_t^{-1}(\sigma_t(s))\,|$",
    ylabel=r"$|\,s - \sigma_t^{-1}(\sigma_t(s))\,|$",
    label=r"roundtrip error",
    paper_ready=False,
    legend_loc="best",
    save_path=None,
    show=True,
):
    """Plot absolute error of a warp roundtrip map on the time domain."""
    config = get_plot_config(paper_ready)
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
        plt.title(title or "Roundtrip error", fontsize=config["axis_label_fontsize"])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=config["legend_fontsize"], loc=legend_loc)
    _finalize_figure(save_path=save_path, show=show)


def plot_inner_product_matrix(
    inner_mat,
    title="Inner-product matrix",
    paper_ready=False,
    save_path=None,
    show=True,
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
    _finalize_figure(save_path=save_path, show=show)
