# %%
import math
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
import numpy.polynomial.legendre as leg
from research_hippo_with_products.utils import find_project_root
from research_hippo_with_products.utils.functions import (
    compare_stat_signal,
    sigma,
    sigma_inverse,
    sigma_derivative,
    compute_basis,
    compute_coefficients,
)

from research_hippo_with_products.visualizers import (
    plot_basis_functions,
    plot_reconstructed_signal,
    plot_signal,
    plot_reconstruction_error,
    select_basis_for_plot,
)


def run_ssm(A, B, c_init, input_signal, scale=0.0030025):
    c_t = c_init
    for f in input_signal:
        c_t_plus_1 = A @ c_t + B * f * scale
        c_t = c_t_plus_1
    return c_t


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

    tau = 1  # Scale parameter

    # Process whitenoise and Lorenz63
    whitenoise_half_index = len(whitenoise_data["time"]) // 2
    whitenoise_t_ref = whitenoise_data["time"][whitenoise_half_index]
    whitenoise_train_time = whitenoise_data["time"][:whitenoise_half_index]  # s_i
    whitenoise_time = whitenoise_data["time"]  # s_i
    whitenoise_train_signal = np.asarray(
        whitenoise_data["data"][:whitenoise_half_index], dtype=np.complex128
    )  # f_i in the original domain
    whitenoise_signal = np.asarray(
        whitenoise_data["data"], dtype=np.float64
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
        # ks = np.arange(1000)

    plot_signal(
        whitenoise_train_time,
        whitenoise_train_signal,
        "Whitenoise",
        "s",
        plot_markers=False,
    )

    time = whitenoise_time
    signal = whitenoise_signal
    start_index = whitenoise_half_index
    ks = ks
    tau = tau
    basis_type = basis_type
    data_name = "Whitenoise"
    recon_domain = "s"

    # %%
    L_bases = len(time[:start_index])
    z_vals_t = sigma(time[:start_index], tau)
    z_vals_t_plus_1 = sigma(time[: start_index + 1], tau)
    weights_t_plus_1 = sigma_derivative(time[: start_index + 1], tau)
    # %%
    import torch
    import math

    def hippo_legt_A_B(K, dtype=torch.float64):
        n = torch.arange(K, dtype=dtype)
        sqrt_2np1 = torch.sqrt(2 * n + 1)
        A = torch.zeros((K, K), dtype=dtype)
        idx = torch.arange(K)
        A[idx, idx] = -(2 * n + 1)  # diagonal
        idx_off = torch.arange(K - 1)
        A[idx_off, idx_off + 1] = sqrt_2np1[:-1] * sqrt_2np1[1:]  # superdiag
        A[idx_off + 1, idx_off] = sqrt_2np1[1:] * sqrt_2np1[:-1]  # subdiag
        B = sqrt_2np1.unsqueeze(1) * (1 - math.exp(-1))  # ZOH scale
        return A, B

    K = len(ks)
    A, B = hippo_legt_A_B(K)

    # %%
    # # Compute A & B from uniform z values in range [-1, 1]
    # uniform_A_integrand_size = 1000
    # uniform_z_vals_t_plus_1 = np.linspace(
    #     -1, 1, uniform_A_integrand_size + 1, endpoint=False
    # )
    # uniform_z_vals_t_plus_1 = uniform_z_vals_t_plus_1 + (
    #     uniform_z_vals_t_plus_1[1] - uniform_z_vals_t_plus_1[0]
    # )
    # t_from_uniform_z_vals_t_plus_1 = sigma_inverse(
    #     uniform_z_vals_t_plus_1, time[start_index], tau
    # )
    # uniform_z_vals_t = sigma(t_from_uniform_z_vals_t_plus_1[:-1], tau)
    # weights_t_plus_1_from_uniform_z_vals = sigma_derivative(
    #     t_from_uniform_z_vals_t_plus_1, tau
    # )
    # Bases_t_from_uniform_z_vals = compute_basis(uniform_z_vals_t, ks, basis_type)
    # Bases_t_extended_t_plus_1_from_uniform_z_vals = np.concatenate(
    #     (Bases_t_from_uniform_z_vals, np.zeros((1, len(ks)))),
    #     axis=0,
    # )
    # Bases_t_plus_1_from_uniform_z_vals = compute_basis(
    #     uniform_z_vals_t_plus_1, ks, basis_type
    # )
    #
    # A_from_uniform_z_vals, bases_t_plus_1_from_uniform_z_vals = compute_coefficients(
    #     Bases_t_extended_t_plus_1_from_uniform_z_vals,
    #     t_from_uniform_z_vals_t_plus_1,
    #     uniform_z_vals_t_plus_1,
    #     ks,
    #     weights_t_plus_1_from_uniform_z_vals,
    #     basis_type=basis_type,
    # )
    # B_from_uniform_z_vals = bases_t_plus_1_from_uniform_z_vals[-1]

    # A, B = A_from_uniform_z_vals, B_from_uniform_z_vals
    # bases_t_plus_1 = bases_t_plus_1_from_uniform_z_vals
    # %%
    # Compute bases for range (-inf, index]
    Bases_t = compute_basis(z_vals_t, ks, basis_type)
    # Extend bases (-inf, index] to domain: (-inf, index + 1]
    Bases_t_extended_t_plus_1 = np.concatenate(
        (Bases_t, np.zeros((1, len(ks)))),
        axis=0,
    )
    # Compute bases for range (-inf, index + 1]
    Bases_t_plus_1 = compute_basis(z_vals_t_plus_1, ks, basis_type)
    print("Bases_t shape:", Bases_t.shape)
    print("Bases_t_extended_t_plus_1 shape:", Bases_t_extended_t_plus_1.shape)
    print("Bases_t_plus_1 shape:", Bases_t_plus_1.shape)

    A, bases_t_plus_1 = compute_coefficients(
        Bases_t_extended_t_plus_1,
        time[: start_index + 1],
        z_vals_t_plus_1,
        ks,
        weights_t_plus_1,
        basis_type=basis_type,
    )

    B = bases_t_plus_1[-1]
    # %%
    time_to_plot = time[: start_index + 1]
    reconstructed_basis_t_extended_t_plus_1 = np.conjugate(bases_t_plus_1) @ A
    plot_basis_functions(
        x=time[: start_index + 1],
        Bases=reconstructed_basis_t_extended_t_plus_1,
        ks=ks,
        data_name=data_name,
        domain_name="s",
        basis_type=basis_type,
        title=f"Reconstructed basis functions for t in [0, {time_to_plot[-1]}]",
    )

    # %%
    L_scan = 10

    signal_to_scan = signal[start_index : start_index + L_scan]
    Bases_t_plus_L_scan = compute_basis(
        sigma(time[: start_index + L_scan], tau),
        ks,
        basis_type,
    )

    # %%
    ########## test c_init = 0 ##########
    print("# Testing with c_init = 0")
    c_init = np.zeros(len(ks), dtype=np.float32)
    run_ssm_result = run_ssm(A, B, c_init, signal_to_scan)
    print("Run SSM Result Shape:", run_ssm_result.shape)
    print("run signal_to_scan.shape:", signal_to_scan.shape)

    # Reconstruct the signal
    reconstructed_signal = Bases_t_plus_L_scan @ run_ssm_result
    print("Reconstructed Signal Shape:", reconstructed_signal.shape)
    print("Original scan signal values:", signal_to_scan)
    print(
        "Reconstructed scan signal values:",
        reconstructed_signal[start_index : start_index + L_scan],
    )

    plot_reconstructed_signal(
        time[: start_index + L_scan],
        signal[: start_index + L_scan],
        reconstructed_signal,
        data_name,
        recon_domain,
    )
    plot_reconstructed_signal(
        time[start_index - 20 : start_index + L_scan],
        signal[start_index - 20 : start_index + L_scan],
        reconstructed_signal[start_index - 20 : start_index + L_scan],
        data_name,
        recon_domain,
    )

    # %%
    ### Testing with c_init = c_t [0:index] ##########
    print("# Testing with c_init = c_t [0:index]")
    c_init, _ = compute_coefficients(
        signal[:start_index],
        time[:start_index],
        z_vals_t,
        ks,
        np.ones(L_bases),
        basis_type=basis_type,
    )
    run_ssm_result = run_ssm(A, B, c_init, signal_to_scan)
    print("Run SSM Result Shape with c_init:", run_ssm_result.shape)
    reconstructed_signal = Bases_t_plus_L_scan @ run_ssm_result
    print("Reconstructed Signal Shape with c_init:", reconstructed_signal.shape)
    print("Original scan signal values:", signal_to_scan)
    print(
        "Reconstructed scan signal values:",
        reconstructed_signal[start_index : start_index + L_scan],
    )

    plot_reconstructed_signal(
        time[: start_index + L_scan],
        signal[: start_index + L_scan],
        reconstructed_signal,
        data_name,
        recon_domain,
    )
    plot_reconstructed_signal(
        time[start_index - 20 : start_index + L_scan],
        signal[start_index - 20 : start_index + L_scan],
        reconstructed_signal[start_index - 20 : start_index + L_scan],
        data_name,
        recon_domain,
    )


# %%
# Original scan signal values: [ 0.         -0.10667586]
# Reconstructed scan signal values: [-0.00162929 -0.10657069]
