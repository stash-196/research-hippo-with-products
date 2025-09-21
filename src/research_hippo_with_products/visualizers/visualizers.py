import numpy as np
import matplotlib.pyplot as plt


class PlottingConfig:
    """
    A configuration class to hold shared settings for plotting functions.
    Instantiate this class with paper_ready=True for publication-quality figures.
    """

    def __init__(self, paper_ready=False):
        self.paper_ready = paper_ready
        if self.paper_ready:
            # Settings for a publication-quality figure ✒️
            self.axis_label_fontsize = 20
            self.tick_label_fontsize = 16
            self.legend_fontsize = 16
            self.line_width = 2.5
            self.marker_size = 8
            # Use a font that is common in academic papers
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams[
                "font.serif"
            ]
        else:
            # Default settings for exploratory plotting
            self.axis_label_fontsize = None  # Matplotlib default
            self.tick_label_fontsize = None
            self.legend_fontsize = None
            self.line_width = 1.5
            self.marker_size = 6
            # plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["Arial"] + plt.rcParams[
                "font.sans-serif"
            ]


def plot_reconstructed_signal(
    x, original, reconstructed, data_name, domain_name="s", title=""
):
    plt.figure(figsize=(10, 6))
    plt.plot(x, original, label=f"Original {data_name} f({domain_name})")
    plt.plot(
        x,
        np.real(reconstructed),
        label=f"Reconstructed {data_name} f({domain_name})",
        linestyle="--",
        color="red",
    )
    plt.xlabel(f"{domain_name}")
    plt.ylabel(f"{data_name} f({domain_name})")
    plt.title(title + f"Reconstruction of {data_name} in {domain_name} domain")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_signal(
    x, signal, data_name, domain_name, plot_complex=False, plot_markers=True
):
    plt.figure(figsize=(10, 6))
    if plot_complex:
        plt.plot(x, np.real(signal), label=f"Re[F({domain_name})]")
        plt.plot(x, np.imag(signal), label=f"Im[F({domain_name})]", linestyle="--")
    else:
        if plot_markers:
            # plot in marker, no lines, marker empty inside
            plt.plot(
                x,
                signal,
                label=f"F({domain_name})",
                marker="o",
                linestyle="",
                fillstyle="none",
            )
        else:
            plt.plot(x, signal, label=f"F({domain_name})", linestyle="-")
    plt.xlabel(f"{domain_name}")
    plt.ylabel(f"F({domain_name})")
    plt.title(f"{data_name} in {domain_name} domain")
    plt.grid(True)
    plt.legend()
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
):
    """
    Plot a subset of basis functions against the domain, including
    the first and last few functions.

    Parameters:
    - x: Domain values (s or z)
    - Bases: Basis matrix (columns correspond to basis functions)
    - ks: Fourier indices or polynomial degrees
    - data_name: Name of the dataset (e.g., "Whitenoise")
    - domain_name: "s" or "z"
    - basis_type: "fourier" or "polynomial"
    - num_functions_front: Number of basis functions to plot from the start
    - num_functions_back: Number of basis functions to plot from the end
    """
    plt.figure(figsize=(12, 8))
    N = Bases.shape[1]
    print("num to vis", num_functions_back, num_functions_front)
    if num_functions_front is None and num_functions_back is None:
        selected_indices = np.arange(N)
        print("printing all basis functions")
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
        if basis_type == "fourier":
            label = f"φ_{k}({domain_name})"
            plt.plot(x, Bases[:, i], label=label)
        elif basis_type == "polynomial":
            label = f"φ_{k}({domain_name})"
            plt.plot(x, Bases[:, i], label=label)
        elif basis_type == "legendre":
            label = f"P_{k}({domain_name})"
            plt.plot(x, Bases[:, i], label=label)

    plt.xlabel(f"{domain_name}")
    plt.ylabel("Basis Function Value")
    plt.title(
        title
        + f"{data_name} Basis Functions in {domain_name} Domain ({basis_type.capitalize()})\n"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plot reconstruction error
def plot_reconstruction_error(original, reconstructed, data_name, domain_name="s"):
    """
    Plot the reconstruction error between the original and reconstructed signals.

    Parameters:
    - original: Original signal array
    - reconstructed: Reconstructed signal array
    - data_name: Name of the dataset
    - domain_name: "s" or "z"
    """
    error = np.abs(original - reconstructed)
    plt.figure(figsize=(10, 6))
    plt.plot(error, label="Reconstruction Error", color="red")
    plt.xlabel(f"{domain_name}")
    plt.ylabel("Error Magnitude")
    plt.title(f"Reconstruction Error for {data_name} in {domain_name} Domain")
    plt.grid(True)
    plt.legend()
    plt.show()


def select_basis_for_plot(ks, Bases, basis_type):
    if basis_type == "fourier":
        ks_plot = ks[ks >= 0]
        Bases_plot = Bases[:, np.where(ks >= 0)[0]].copy()
        # reverse order of Bases_plot
        Bases_plot = Bases_plot[:, ::-1]
    else:
        ks_plot = ks
        Bases_plot = Bases.copy()
    return ks_plot, Bases_plot
