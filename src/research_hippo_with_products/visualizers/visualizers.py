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
    num_functions_front=1,
    num_functions_back=1,
    num_functions_middle=1,
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

    # Select indices for front and back
    front_indices = np.arange(num_functions_front)
    back_indices = np.arange(len(ks) - num_functions_back, len(ks))
    middle_indices = np.arange(
        len(ks) // 2 - num_functions_middle, len(ks) // 2 + num_functions_middle
    )

    # Combine front and back indices
    selected_indices = np.concatenate((front_indices, back_indices, middle_indices))

    for i in selected_indices:
        k = ks[i]
        if basis_type == "fourier":
            label_real = f"Re[φ_{k}({domain_name})]"
            label_imag = f"Im[φ_{k}({domain_name})]"
            plt.plot(x, np.real(Bases[:, i]), label=label_real)
            plt.plot(x, np.imag(Bases[:, i]), linestyle="--", label=label_imag)
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
        f"First {num_functions_front}, Middle {num_functions_middle}, and Last {num_functions_back} Functions"
    )
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # Place legend outside plot
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
