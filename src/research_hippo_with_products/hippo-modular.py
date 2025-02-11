# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from research_hippo_with_products.utils import find_project_root


# %%
# Define transformation functions
def zeta(t, s, tau=1):
    return 2 * np.exp((s - t) / tau) - 1


def zeta_inverse(t, z, tau=1):
    return np.log((z + 1) / 2) * tau + t


def zeta_derivative(t, s, tau=1):
    return 2 * np.exp((s - t) / tau) / tau


from scipy.special import legendre


# Plot function
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


# Generalized basis computation
def compute_basis(x, ks, basis_type="fourier"):
    """
    Compute the basis matrix for a given basis type.

    Parameters:
    - x: Array of domain values (s or z)
    - ks: Array of basis indices (e.g., Fourier indices or polynomial degrees)
    - basis_type: Type of basis ("fourier", "polynomial", "legendre")

    Returns:
    - Basis matrix with columns as basis functions evaluated at 'x'
    """
    if basis_type == "fourier":
        # Construct real Fourier basis (excluding DC term for sine)
        Bases_cos = np.cos(np.pi * np.outer(ks, x)).T  # Cosine basis
        Bases_sin = np.sin(
            np.pi * np.outer(ks[1:], x)
        ).T  # Sine basis (excluding DC term)
        # Concatenate all basis functions
        return np.hstack([Bases_cos, Bases_sin])
        # Scale frequencies with pi to match the standardized domain [-1, 1]
        # return np.exp(-1j * np.pi * np.outer(ks, x)).T
    elif basis_type == "legendre":
        return np.array([legendre(k)(x) for k in ks]).T
    else:
        raise ValueError(f"Unsupported basis type: {basis_type}")


# Fourier/polynomial transform and reconstruction
def compute_coefficients(
    signal, time, z_vals, ks, weights, domain_name, basis_type="fourier"
):
    """
    Compute coefficients for a given signal in the z domain using the chosen basis.
    """

    if domain_name == "s":
        Psi = compute_basis(z_vals, ks, basis_type)
        delta_s = time[1] - time[0]  # If uniform
        ck = (signal * weights) @ Psi * delta_s
        # ck = (signal * weights) @ Psi
        return ck, Psi
    elif domain_name == "z":
        Phi = compute_basis(time, ks, basis_type)
        # compute non-uniform delta_z
        delta_z = np.diff(z_vals)
        ck = signal @ Phi * delta_z
        return ck, Phi


def normalize_basis(Bases, ks, basis_type="legendre"):
    """
    Normalize the basis functions based on their type.

    Parameters:
    - Bases: Basis matrix (columns correspond to basis functions)
    - ks: Basis indices
    - basis_type: "fourier", "polynomial", or "legendre"

    Returns:
    - Normalized basis matrix
    """
    if basis_type == "fourier":
        # For Fourier basis, normalization can be handled during basis computation
        # Assuming the basis functions are already normalized
        return Bases
    elif basis_type == "polynomial":
        # Orthogonalize if necessary (e.g., using Gram-Schmidt)
        # For simplicity, returning as is
        return Bases
    elif basis_type == "legendre":
        # Normalize based on Legendre orthogonality
        # Integral of P_k(x)^2 over [-1,1] is 2 / (2k + 1)
        normalization_factors = np.sqrt(2 / (2 * ks + 1))
        return Bases * normalization_factors[:, np.newaxis]
    else:
        return Bases


# Process whitenoise data
def process_signal(
    time, signal, t_ref, K, tau, basis_type, data_name="Whitenoise", domain_name="s"
):
    ks = np.arange(K + 1)
    z_vals = zeta(t_ref, time, tau)
    weights = zeta_derivative(t_ref, time, tau)

    # Compute coefficients
    if domain_name == "s":
        ck, Bases = compute_coefficients(
            signal, time, z_vals, ks, weights, domain_name, basis_type
        )
    elif domain_name == "z":
        ck, Bases = compute_coefficients(
            signal, time, z_vals, ks, weights, domain_name, basis_type
        )

        # Normalize basis if necessary

    if basis_type == "legendre":
        Bases = normalize_basis(Bases, ks, basis_type)

    # Reconstruct signal
    reconstructed = np.conjugate(Bases) @ ck

    # Plot results
    plot_reconstructed_signal(z_vals, signal, reconstructed, data_name, "z")
    plot_reconstructed_signal(time, signal, reconstructed, data_name, "s")

    # if domain_name == "z":
    #     plot_signal(z_vals, signal, data_name, domain_name="z")
    # elif domain_name == "s":
    #     plot_signal(time, signal, data_name, domain_name="s")
    #     plot_signal(time, weights, "w", domain_name="s")

    # Plot basis functions
    plot_basis_functions(
        x=time,
        Bases=Bases,
        ks=ks,
        data_name=data_name,
        domain_name=domain_name,
        basis_type=basis_type,
    )

    # Plot reconstruction error
    # plot_reconstruction_error(signal, reconstructed, data_name, domain_name)


# Load data
file_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.join(file_dir, "..", ".."))
data_dir = os.path.join(root_dir, "data")

whitenoise_data_path = os.path.join(data_dir, "whitenoise_samples.pkl")
lorenz63_data_path = os.path.join(data_dir, "lorenz63_samples.pkl")

with open(whitenoise_data_path, "rb") as f:
    whitenoise_data = pickle.load(f)

with open(lorenz63_data_path, "rb") as f:
    lorenz63_data = pickle.load(f)

# Parameters
K = 1000  # Number of Fourier basis functions
tau = 2  # Scale parameter

# Process whitenoise and Lorenz63
whitenoise_half_point = len(whitenoise_data["time"]) // 2
whitenoise_t_ref = whitenoise_data["time"][whitenoise_half_point]
whitenoise_train_time = whitenoise_data["time"][:whitenoise_half_point]  # s_i
whitenoise_train_data = np.asarray(
    whitenoise_data["data"][:whitenoise_half_point], dtype=np.complex128
)  # f_i in the original domain

lorenz63_half_point = len(lorenz63_data["time"]) // 2
lorenz63_t_ref = lorenz63_data["time"][lorenz63_half_point]
lorenz63_train_time = lorenz63_data["time"][:lorenz63_half_point]  # s_i
lorenz63_train_data = np.asarray(lorenz63_data["data"], dtype=np.complex128)[
    :lorenz63_half_point,
    0,
]  # f_i in the original domain

process_signal(
    whitenoise_train_time,
    whitenoise_train_data,
    whitenoise_t_ref,
    K,
    tau,
    "fourier",
    "Whitenoise",
    "s",
)


# process_signal(
#     lorenz63_train_time,
#     lorenz63_train_data,
#     lorenz63_t_ref,
#     ks,
#     tau,
#     "fourier",
#     "Lorenz63",
#     "z",
# )
# %%
