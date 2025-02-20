import numpy as np
import matplotlib.pyplot as plt


def plot_reconstructed_signal(x, original, reconstructed, data_name, domain_name="s"):
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
    plt.title(f"Original vs Reconstructed {data_name} in {domain_name} domain")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_signal(x, signal, data_name, domain_name):
    plt.figure(figsize=(10, 6))
    plt.plot(x, np.real(signal), label=f"Re[F({domain_name})]")
    plt.plot(x, np.imag(signal), label=f"Im[F({domain_name})]", linestyle="--")
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
        f"{data_name} Basis Functions in {domain_name} Domain ({basis_type.capitalize()})\n"
        f"First {num_functions_front}, and Last {num_functions_back} Functions"
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
