# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
import numpy.polynomial.legendre as leg
from research_hippo_with_products.utils import find_project_root
from research_hippo_with_products.utils.utils import compare_stat_signal
from research_hippo_with_products.visualizers import (
    plot_basis_functions,
    plot_reconstructed_signal,
    plot_signal,
    plot_reconstruction_error,
)
from research_hippo_with_products.hippo_modular import (
    zeta,
    zeta_inverse,
    zeta_derivative,
)


def compute_basis(x, ks, basis_type="fourier"):
    if basis_type == "fourier":
        return np.exp(-1j * 2 * np.pi * np.outer(ks, x)).T
    elif basis_type == "legendre":
        return np.array([legendre(k)(x) for k in ks]).T
    else:
        raise ValueError(f"Unsupported basis type: {basis_type}")


# Fourier/polynomial transform and reconstruction
def compute_coefficients(
    signal, time, z_vals, ks, weights, basis_type="fourier", verbose=False
):
    Psi = compute_basis(z_vals, ks, basis_type)
    if basis_type == "fourier":
        ck = (signal * weights) @ Psi / len(time)
        return ck, Psi
    elif basis_type == "legendre":
        # if signal is 1D
        if len(signal.shape) == 1:
            ck = leg.legfit(z_vals, signal, len(ks) - 1)
        else:
            print("signal shape", signal[:, 0].shape, z_vals.shape)
            ck = np.array(
                [
                    leg.legfit(z_vals, signal[:, i], len(ks) - 1)
                    for i in range(signal.shape[1])
                ]
            ).T
        return ck, Psi


def process_signal(
    train_time,
    train_signal,
    time,
    signal,
    train_index,
    t_ref,
    ks,
    tau,
    basis_type,
    data_name="Whitenoise",
    recon_domain="s",
):
    index = train_index
    z_vals_t = zeta(time[index], time[:index], tau)
    weights_t = zeta_derivative(time[index], time[:index], tau)
    ck_t, Bases_t = compute_coefficients(
        signal[:index], time[:index], z_vals_t, ks, weights_t, basis_type
    )

    z_vals_t_plus_1 = zeta(time[index + 1], time[: index + 1], tau)
    # add a 0 element to the end of z_vals_t
    z_vals_t_to_t_plus_1 = np.append(z_vals_t, 0)

    Bases_t_plus_1 = compute_basis(z_vals_t_plus_1, ks, basis_type)
    Bases_t_to_t_plus_1 = compute_basis(z_vals_t_to_t_plus_1, ks, basis_type)

    A = np.conjugate(Bases_t_to_t_plus_1.T) @ Bases_t_plus_1

    # create zero array with last element as 1
    delta_t_plus_1 = np.zeros(len(Bases_t_plus_1), dtype=np.complex128)
    delta_t_plus_1[-1] = 1
    print("delta_t_plus_1", delta_t_plus_1.shape)
    b = Bases_t_plus_1.T @ delta_t_plus_1

    ck_t_plus_1 = A @ ck_t + b
    reconstructed_plus_1_ssm = np.conjugate(Bases_t_plus_1) @ ck_t_plus_1

    print("reconstructed_plus_1", reconstructed_plus_1_ssm.shape)
    plot_reconstructed_signal(
        time[: index + 1],
        signal[: index + 1],
        reconstructed_plus_1_ssm,
        data_name,
        recon_domain,
        f"SSM[0,t+1] ",
    )

    # Compute ck_t+n
    n = 20
    z_plus_n = zeta(time[train_index + n], time[: train_index + n], tau)
    weights_plus_n = zeta_derivative(
        time[train_index + n], time[: train_index + n], tau
    )
    ck_t_plus_n, Bases_t_plus_n = compute_coefficients(
        signal[: train_index + n],
        time[: train_index + n],
        z_plus_n,
        ks,
        weights_plus_n,
        basis_type,
    )
    reconstructed_plus_1 = np.conjugate(Bases_t_plus_n) @ ck_t_plus_n
    plot_reconstructed_signal(
        time[: train_index + n],
        signal[: train_index + n],
        reconstructed_plus_1,
        data_name,
        recon_domain,
        f"[0,t+{n}] ",
    )


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
    tau = 2  # Scale parameter

    # Process whitenoise and Lorenz63
    whitenoise_half_index = len(whitenoise_data["time"]) // 2
    whitenoise_t_ref = whitenoise_data["time"][whitenoise_half_index]
    whitenoise_train_time = whitenoise_data["time"][:whitenoise_half_index]  # s_i
    whitenoise_time = whitenoise_data["time"]  # s_i
    whitenoise_train_signal = np.asarray(
        whitenoise_data["data"][:whitenoise_half_index], dtype=np.complex128
    )  # f_i in the original domain
    whitenoise_signal = np.asarray(
        whitenoise_data["data"], dtype=np.complex128
    )  # f_i in the original domain
    # set K based on nyquist
    basis_type = "fourier"
    basis_type = "legendre"
    N_train = len(whitenoise_train_time)
    T_train = whitenoise_train_time[1] - whitenoise_train_time[0]
    if basis_type == "fourier":
        ks = np.arange(-N_train // 2 + 1, N_train // 2) / N_train / T_train
    elif basis_type == "legendre":
        ks = np.arange(N_train // 3)
        ks = np.arange(10)

    process_signal(
        whitenoise_train_time,
        whitenoise_train_signal,
        whitenoise_time,
        whitenoise_signal,
        whitenoise_half_index,
        whitenoise_t_ref,
        ks,
        tau,
        basis_type,
        "Whitenoise",
        "s",
    )


# %%
