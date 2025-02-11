# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from research_hippo_with_products.utils import find_project_root
from research_hippo_with_products.visualizers import (
    plot_basis_functions,
    plot_reconstructed_signal,
    plot_signal,
    plot_reconstruction_error,
)


# %%
# Define transformation functions
def zeta(t, s, tau=1):
    return 2 * np.exp((s - t) / tau) - 1


def zeta_inverse(t, z, tau=1):
    return np.log((z + 1) / 2) * tau + t


def zeta_derivative(t, s, tau=1):
    return 2 * np.exp((s - t) / tau) / tau


from scipy.special import legendre


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
root_dir = find_project_root(__file__)
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
