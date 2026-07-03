import numpy as np
import matplotlib.pyplot as plt


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

    if not paper_ready:
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
        loc="upper center",  # Or "best", etc.
        bbox_to_anchor=(0.5, 1.1),  # Place above the plot, centered
        ncol=2,  # Arrange in 2 columns
    )
    plt.tight_layout()
    plt.show()


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
        k = ks[i]
        if basis_type == "legendre":
            label = f"$P_{k}({domain_name})$"  # Using LaTeX for legendre
        else:  # fourier, polynomial
            label = f"$\phi_{k}({domain_name})$"  # Using LaTeX for phi
        plt.plot(x, Bases[:, i], label=label, linewidth=config["line_width"])

    # --- Apply text size, labels, and conditional title ---
    plt.xlabel(f"{domain_name}", fontsize=config["axis_label_fontsize"])
    plt.ylabel("Basis Function Value", fontsize=config["axis_label_fontsize"])
    plt.xticks(fontsize=config["tick_label_fontsize"])
    plt.yticks(fontsize=config["tick_label_fontsize"])

    if not paper_ready:
        full_title = (
            (title + " " if title else "")
            + f"{data_name} Basis Functions in {domain_name} Domain ({basis_type.capitalize()})"
        )
        plt.title(full_title, fontsize=config["axis_label_fontsize"])

    plt.grid(True, linestyle="--", alpha=0.6)
    # plt.legend(fontsize=config["legend_fontsize"])
    plt.tight_layout()
    plt.show()
