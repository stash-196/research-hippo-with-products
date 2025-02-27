# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from research_hippo_with_products.utils import find_project_root
from research_hippo_with_products.utils.utils import compare_stat_signal
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
        # N = len(x)
        # # Construct real Fourier basis (excluding DC term for sine)
        # Bases_cos = np.cos(np.pi * np.outer(ks, x)).T / np.sqrt(N)  # Cosine basis
        # Bases_sin = np.sin(np.pi * np.outer(ks[1:], x)).T / np.sqrt(
        #     N
        # )  # Sine basis (excluding DC term)
        # # Concatenate all basis functions
        # return np.hstack([Bases_cos, Bases_sin])
        # Scale frequencies with pi to match the standardized domain [-1, 1] and normalize
        return np.exp(-1j * 2 * np.pi * np.outer(ks, x)).T
    elif basis_type == "legendre":
        return np.array([legendre(k)(x) for k in ks]).T
    else:
        raise ValueError(f"Unsupported basis type: {basis_type}")


# Fourier/polynomial transform and reconstruction
def compute_coefficients(
    signal, time, z_vals, ks, weights, domain_name, basis_type="fourier", verbose=False
):
    """
    Compute coefficients for a given signal in the z domain using the chosen basis.
    """

    if domain_name == "s":
        Psi = compute_basis(z_vals, ks, basis_type)
        if basis_type == "fourier":
            delta_s = time[1] - time[0] if len(time) > 1 else 1
            T = time[-1] - time[0]
            if verbose:
                print("T: ", T)
                print("delta_s: ", delta_s)
                print("len(time): ", len(time))
                print("T/delta_s vs len(time): ", T / delta_s, " : ", len(time))
                print("1/N: ", 1 / len(time))
                print("weights: ", weights)
            ck = (signal * weights) @ Psi / len(time)
            # ck = (signal * weights) @ Psi
            return ck, Psi
        elif basis_type == "legendre":
            import numpy.polynomial.legendre as leg

            ck = leg.legfit(z_vals, signal, len(ks) - 1)
            return ck, Psi
    elif domain_name == "z":
        Phi = compute_basis(time, ks, basis_type)
        # compute non-uniform delta_z
        delta_z = np.diff(z_vals)
        ck = signal @ Phi * delta_z
        return ck, Phi


# Process whitenoise data
def process_signal(
    time, signal, t_ref, ks, tau, basis_type, data_name="Whitenoise", recon_domain="s"
):
    z_vals = zeta(t_ref, time, tau)
    weights = zeta_derivative(t_ref, time, tau)

    # Compute coefficients
    if recon_domain == "s":
        ck, Bases = compute_coefficients(
            signal, time, z_vals, ks, weights, recon_domain, basis_type
        )
    elif recon_domain == "z":
        ck, Bases = compute_coefficients(
            signal, time, z_vals, ks, weights, recon_domain, basis_type
        )

    # Reconstruct signal
    reconstructed = np.conjugate(Bases) @ ck

    # Plot results
    plot_reconstructed_signal(z_vals, signal, reconstructed, data_name, "z")
    plot_reconstructed_signal(time, signal, reconstructed, data_name, "s")

    # Plot basis functions
    if basis_type == "fourier":
        ks_plot = ks[ks >= 0]
        Bases_plot = Bases[:, np.where(ks >= 0)[0]].copy()
        # reverse order of Bases_plot
        Bases_plot = Bases_plot[:, ::-1]
    else:
        ks_plot = ks
        Bases_plot = Bases.copy()
    plot_basis_functions(
        x=time,
        Bases=Bases_plot,
        ks=ks_plot,
        data_name=data_name,
        domain_name=recon_domain,
        basis_type=basis_type,
    )
    plot_basis_functions(
        x=z_vals,
        Bases=Bases_plot,
        ks=ks_plot,
        data_name=data_name,
        domain_name=recon_domain,
        basis_type=basis_type,
    )
    compare_stat_signal(signal, reconstructed, data_name, "Reconstructed")

    # Plot reconstruction error
    # plot_reconstruction_error(signal, reconstructed, data_name, domain_name)


if __name__ == "__main__":
    # Load data
    root_dir = find_project_root(__file__)
    data_dir = os.path.join(root_dir, "data")

    whitenoise_data_path = os.path.join(data_dir, "whitenoise_samples.pkl")
    lorenz63_data_path = os.path.join(data_dir, "lorenz63_samples.pkl")

    with open(whitenoise_data_path, "rb") as f:
        whitenoise_data = pickle.load(f)

    with open(lorenz63_data_path, "rb") as f:
        lorenz63_data = pickle.load(f)

    # %%
    # Parameters
    # nyquist
    tau = 1  # Scale parameter

    # Process whitenoise and Lorenz63
    whitenoise_half_point = len(whitenoise_data["time"]) // 2
    whitenoise_t_ref = whitenoise_data["time"][whitenoise_half_point]
    whitenoise_train_time = whitenoise_data["time"][:whitenoise_half_point]  # s_i
    whitenoise_train_data = np.asarray(
        whitenoise_data["data"][:whitenoise_half_point], dtype=np.complex128
    )  # f_i in the original domain
    whitenoise_delta_t = whitenoise_train_time[1] - whitenoise_train_time[0]
    whitenoise_nyquist = 1 / (2 * whitenoise_delta_t)
    # set K based on nyquist
    basis_type = "legendre"
    basis_type = "fourier"
    N = len(whitenoise_train_time)
    T = whitenoise_train_time[1] - whitenoise_train_time[0]
    if basis_type == "fourier":
        ks = np.arange(-N // 2 + 1, N // 2) / N / T
    elif basis_type == "legendre":
        ks = np.arange(N // 3)
    # K = 40

    process_signal(
        whitenoise_train_time,
        whitenoise_train_data,
        whitenoise_t_ref,
        ks,
        tau,
        basis_type,
        "Whitenoise",
        "s",
    )

    # lorenz63_half_point = len(lorenz63_data["time"]) // 2
    # lorenz63_t_ref = lorenz63_data["time"][lorenz63_half_point]
    # lorenz63_train_time = lorenz63_data["time"][:lorenz63_half_point]  # s_i
    # lorenz63_train_data = np.asarray(lorenz63_data["data"], dtype=np.complex128)[
    #     :lorenz63_half_point,
    #     0,
    # ]  # f_i in the original domain

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
