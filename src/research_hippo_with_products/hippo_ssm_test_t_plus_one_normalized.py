# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre, eval_legendre
import numpy.polynomial.legendre as leg
from research_hippo_with_products.utils import find_project_root
from research_hippo_with_products.utils.functions import (
    compare_stat_signal,
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

PAPER_READY_PLOTS = True


def sigma(s: np.ndarray, tau=1):
    t = (
        s[-1]
        if isinstance(s, np.ndarray)
        else ValueError("Input s must be an array-like structure.")
    )
    return 2 * np.exp((s - t) / tau) - 1


def sigma_derivative(s: np.ndarray, tau=1):
    t = (
        s[-1]
        if isinstance(s, np.ndarray)
        else ValueError("Input s must be an array-like structure.")
    )
    return 2 * np.exp((s - t) / tau) / tau


# def sigma(s, tau=None, warp_type="exp", t=None, theta=None):
#     if t is None:
#         t = s[-1]  # current time
#     if warp_type == "linear":  # LegS/LegT
#         s_min = 0 if theta is None else t - theta  # for LegT fixed window
#         return 2 * (s - s_min) / (t - s_min) - 1  # z in [-1,1]
#     elif warp_type == "exp":  # Lag/LegS-exp approx
#         return 2 * np.exp((s - t) / tau) - 1
#     else:
#         raise ValueError


# def sigma_derivative(s, tau=None, warp_type="exp", t=None, theta=None):
#     if t is None:
#         t = s[-1]
#     if warp_type == "linear":
#         length = t - (0 if theta is None else t - theta)
#         return np.full_like(s, 2 / length)  # constant Jacobian
#     elif warp_type == "exp":
#         return (2 / tau) * np.exp((s - t) / tau)
#     else:
#         raise ValueError


# def normalized_legendre(k, x):
#     return np.sqrt((2 * k + 1) / 2.0) * eval_legendre(k, x)  # float for precision


# def compute_basis(x, ks, basis_type="fourier"):
#     if basis_type == "fourier":
#         return np.exp(-1j * 2 * np.pi * np.outer(ks, x)).T
#     elif basis_type == "legendre":
#         # return np.array([normalized_legendre(k, x) for k in ks]).T  # Normalized!
#         return np.array([legendre(k)(x) for k in ks]).T
#     else:
#         raise ValueError(f"Unsupported basis type: {basis_type}")


# # Fourier/polynomial transform and reconstruction
# def compute_coefficients(
#     signal: np.ndarray,
#     time: np.ndarray,
#     z_vals: np.ndarray,
#     ks: np.ndarray,
#     weights: np.ndarray,
#     basis_type: str = "fourier",
#     verbose: bool = False,
# ) -> tuple:
#     """ "" """
#     Psi = compute_basis(z_vals, ks, basis_type)
#     if basis_type == "fourier":
#         T = len(time)
#         ck = (signal * weights) @ Psi / (T)
#         return ck, Psi
#     elif basis_type == "legendre":
#         # if signal is 1D
#         if len(signal.shape) == 1:
#             ck = leg.legfit(z_vals, signal, len(ks) - 1)
#             # weights_normalized = weights / np.sum(weights)
#             # T = len(z_vals)
#             # ck = (signal * weights_normalized) @ Psi
#         else:
#             print("signal shape", signal[:, 0].shape, z_vals.shape)
#             ck = np.array(
#                 [
#                     leg.legfit(z_vals, signal[:, i], len(ks) - 1)
#                     for i in range(signal.shape[1])
#                 ]
#             ).T
#         return ck, Psi
#     else:
#         raise ValueError(f"Unsupported basis type: {basis_type}")


# def main(
#     time,
#     signal,
#     train_index,
#     t_ref,
#     ks,
#     tau,
#     basis_type,
#     data_name="Whitenoise",
#     recon_domain="s",
# ):

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

    tau = 1.0  # Scale parameter

    # Process whitenoise and Lorenz63
    whitenoise_half_index = len(whitenoise_data["time"]) // 2 + 1
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
    # basis_type = "fourier"
    basis_type = "legendre"
    N_train = len(whitenoise_train_time)
    T_train = whitenoise_train_time[1] - whitenoise_train_time[0]
    if basis_type == "fourier":
        ks = np.arange(-N_train // 2 + 1, N_train // 2) / N_train / T_train
    elif basis_type == "legendre":
        ks = np.arange(N_train // 3)
        # ks = np.arange(100)

    plot_signal(
        whitenoise_train_time,
        whitenoise_train_signal,
        "Whitenoise",
        "s",
        function_name="u",
        plot_markers=False,
        paper_ready=PAPER_READY_PLOTS,
    )
    plot_signal(
        whitenoise_time,
        whitenoise_signal,
        "Whitenoise",
        "s",
        plot_markers=False,
        paper_ready=PAPER_READY_PLOTS,
    )

    time = whitenoise_time
    signal = whitenoise_signal
    train_index = whitenoise_half_index
    t_ref = whitenoise_t_ref
    ks = ks
    tau = tau
    basis_type = basis_type
    data_name = "Whitenoise"
    recon_domain = "s"

    # %%
    index = train_index
    z_vals_t = sigma(time[:index], tau)
    weights_t = sigma_derivative(time[:index], tau)
    ck_t, Bases_t = compute_coefficients(
        signal[:index], time[:index], z_vals_t, ks, weights_t, basis_type
    )

    print("index :", index)
    print("z_vals_t", z_vals_t.shape)
    print("weights_t", weights_t.shape)
    print("Bases_t", Bases_t.shape)
    print("ck_t", ck_t.shape)
    plot_basis_functions(
        x=z_vals_t,
        Bases=Bases_t,
        ks=ks,
        data_name=data_name,
        domain_name="z",
        basis_type=basis_type,
        title=f"Phi_[0,t] ",
        paper_ready=PAPER_READY_PLOTS,
    )

    # Test reconstruction with ck_t
    reconstructed_signal_t = np.conjugate(Bases_t) @ ck_t
    print("reconstructed_signal_t", reconstructed_signal_t.shape)
    plot_reconstructed_signal(
        time[:index],
        signal[:index],
        reconstructed_signal_t,
        data_name=data_name,
        domain_name=recon_domain,
        title=f"[0,t] ",
        paper_ready=PAPER_READY_PLOTS,
    )
    plot_reconstructed_signal(
        z_vals_t,
        signal[:index],
        reconstructed_signal_t,
        data_name=data_name,
        domain_name="z",
        function_name="U",
        title=f"[-1,1] ",
        paper_ready=PAPER_READY_PLOTS,
    )
    # plot coefficients
    plot_reconstructed_signal(
        ks,
        ck_t,
        np.zeros_like(ck_t),
        data_name,
        "n",
        f"ck_t[0,t] ",
        paper_ready=PAPER_READY_PLOTS,
    )
    # %%
    # new z_vals and bases in (-inf, t_ref + 1]
    z_vals_t_plus_1 = sigma(time[: index + 1], tau)
    weights_t_plus_1 = sigma_derivative(time[: index + 1], tau)
    Bases_t_plus_1 = compute_basis(z_vals_t_plus_1, ks, basis_type)
    print("Bases_t_plus_1", Bases_t_plus_1.shape)

    # extend bases (-inf, t_ref] to domain: (-inf, t_ref + 1]
    Bases_t_extended_t_plus_1 = np.concatenate(
        (Bases_t, np.zeros((1, len(ks)))),
        axis=0,
    )

    # print shape
    print("Bases_t_to_t_plus_1", Bases_t_extended_t_plus_1.shape)
    plot_basis_functions(
        x=time[: index + 1],
        Bases=Bases_t_extended_t_plus_1,
        ks=ks,
        data_name=data_name,
        domain_name="s",
        basis_type=basis_type,
        title=f"Psi_t[0,t+1]",
        paper_ready=PAPER_READY_PLOTS,
    )
    plot_basis_functions(
        x=z_vals_t_plus_1,
        Bases=Bases_t_extended_t_plus_1,
        ks=ks,
        data_name=data_name,
        domain_name="z",
        basis_type=basis_type,
        title=f"Phi_[0,t+1]",
        paper_ready=PAPER_READY_PLOTS,
    )
    plot_basis_functions(
        x=time[: index + 1],
        Bases=Bases_t_plus_1,
        ks=ks,
        data_name=data_name,
        domain_name="s",
        basis_type=basis_type,
        title=f"Psi_t+1[0,t+1] ",
        paper_ready=PAPER_READY_PLOTS,
    )
    plot_basis_functions(
        x=z_vals_t_plus_1,
        Bases=Bases_t_plus_1,
        ks=ks,
        data_name=data_name,
        domain_name="s",
        basis_type=basis_type,
        title=f"Phi_+1[0,t+1]",
        paper_ready=PAPER_READY_PLOTS,
    )

    ############ Use one basis function as example #############

    example_t_basis_signal = Bases_t_extended_t_plus_1[:, 100]
    plot_signal(
        time[: index + 1],
        example_t_basis_signal,
        f"Psi_t,1[0,t+1] ",
        "s",
        paper_ready=PAPER_READY_PLOTS,
    )
    example_coefficient, example_t_plus_1_basis = compute_coefficients(
        example_t_basis_signal,
        time[: index + 1],
        z_vals_t_plus_1,
        ks,
        weights_t_plus_1,
        basis_type=basis_type,
    )

    difference = Bases_t_plus_1 - example_t_plus_1_basis
    print("difference of Psi_t+1,k", np.sum(difference))

    # reconstruct example signal
    reconstructed_example_signal = (
        np.conjugate(example_t_plus_1_basis) @ example_coefficient
    )
    plot_reconstructed_signal(
        time[: index + 1],
        example_t_basis_signal,
        reconstructed_example_signal,
        data_name="",
        function_name="Example Psi_t,n",
        domain_name=recon_domain,
        title=f"[0,t+1] ",
        paper_ready=PAPER_READY_PLOTS,
    )
    ############### Compute transition matrix A ###################
    A, Bases_t_plus_1_shifted = compute_coefficients(
        Bases_t_extended_t_plus_1,  # shape (N, K)
        time[: index + 1],
        z_vals_t_plus_1,
        ks,
        weights_t_plus_1,
        basis_type=basis_type,
    )

    print("A (basis_shifting_coefficients): ", A.shape)
    print("Max amplitude of eigenvalues of A: ", np.max(np.abs(np.linalg.eigvals(A))))
    print("Bases_t_plus_1_shifted", Bases_t_plus_1_shifted.shape)
    # plot basis functions
    plot_basis_functions(
        x=time[: index + 1],
        Bases=Bases_t_plus_1_shifted,
        ks=ks,
        data_name=data_name,
        domain_name="z",
        basis_type=basis_type,
        title=f"Psi_t+1[0,t+1] ",
        paper_ready=PAPER_READY_PLOTS,
    )

    reconstructed_basis_t_extended_t_plus_1 = np.conjugate(Bases_t_plus_1_shifted) @ A
    plot_basis_functions(
        x=time[: index + 1],
        Bases=reconstructed_basis_t_extended_t_plus_1,
        ks=ks,
        data_name=data_name,
        domain_name="s",
        basis_type=basis_type,
        title=f"Reconstructed Psi_t,n[0,t+1] ",
        paper_ready=PAPER_READY_PLOTS,
    )

    difference_baisis = (
        Bases_t_extended_t_plus_1 - reconstructed_basis_t_extended_t_plus_1
    )
    print("difference (avg) of Psi_t,n", np.mean(np.abs(difference_baisis)))

    # create delta_t+1(s) function: zero array with last element as 1?
    delta_t_plus_1 = np.zeros_like(signal[: index + 1])
    delta_t_plus_1[-1] = 1
    print("delta_t_plus_1", delta_t_plus_1.shape)
    b = Bases_t_plus_1_shifted.T @ delta_t_plus_1

    print("b - Psi_t+1,k(t+1): ", np.mean(np.abs(b - Bases_t_plus_1_shifted[-1])))
    print("b - ones(b): ", np.mean(np.abs(b - np.ones(len(ks)))))

    # print shapes
    print(
        "A",
        A.shape,
        "ck_t",
        ck_t.shape,
        "b",
        b.shape,
        "signal(t+1)[: index + 1]",
        signal[: index + 1].shape,
    )
    # %%
    ###### Reconstruct ck_t_plus_1 using SSM ######
    ########### With B
    print("========== Reconstruct ck_t_plus_1 using SSM ===========")
    print("Reconstruct ck_t_plus_1 using SSM WITH B")
    print(f"adding B*u(t+1)= {signal[index]} to A @ ck_t")
    reconstructed_ck_t_plus_1 = np.conjugate(A) @ ck_t + signal[index] * b
    reconstructed_t_plus_1_ssm = (
        np.conjugate(Bases_t_plus_1) @ reconstructed_ck_t_plus_1
    )
    print("reconstructed signal+plus_1 shape", reconstructed_t_plus_1_ssm.shape)
    diff_recon = signal[: index + 1] - reconstructed_t_plus_1_ssm
    print("difference of reconstructions!(avg): ", np.mean(np.abs(diff_recon)))
    print("Original new data point: ", signal[index])
    print("Reconstructed new data point: ", reconstructed_t_plus_1_ssm[index])
    assert reconstructed_t_plus_1_ssm[-1] == reconstructed_t_plus_1_ssm[index]

    plot_reconstructed_signal(
        time[: index + 1],
        signal[: index + 1],
        reconstructed_t_plus_1_ssm,
        data_name=data_name,
        domain_name=recon_domain,
        title=f"SSSM (t+1)",
        paper_ready=PAPER_READY_PLOTS,
    )
    plot_reconstructed_signal(
        time[index - 20 : index + 1],
        signal[index - 20 : index + 1],
        reconstructed_t_plus_1_ssm[index - 20 : index + 1],
        data_name=data_name,
        domain_name=recon_domain,
        title=f"SSM (t+1) (zoomed in)",
        paper_ready=PAPER_READY_PLOTS,
    )

    ########### With B*scale
    print("========== Reconstruct ck_t_plus_1 using SSM ===========")
    print("========== With B*scale ===========")
    scale = 0.0030025
    print("Reconstruct ck_t_plus_1 using SSM WITH B*scale")
    print(f"adding B*u(t+1)*scale= {signal[index]}*{scale} to A @ ck_t")
    reconstructed_ck_t_plus_1_with_scale = (
        np.conjugate(A) @ ck_t + signal[index] * b * scale
    )
    reconstructed_t_plus_1_ssm_with_scale = (
        np.conjugate(Bases_t_plus_1) @ reconstructed_ck_t_plus_1_with_scale
    )
    print(
        "reconstructed signal+plus_1 shape", reconstructed_t_plus_1_ssm_with_scale.shape
    )
    diff_recon = signal[: index + 1] - reconstructed_t_plus_1_ssm_with_scale
    print("difference of reconstructions!(avg): ", np.mean(np.abs(diff_recon)))
    print("Original new data point: ", signal[index])
    print(
        "Reconstructed new data point: ", reconstructed_t_plus_1_ssm_with_scale[index]
    )
    assert (
        reconstructed_t_plus_1_ssm_with_scale[-1]
        == reconstructed_t_plus_1_ssm_with_scale[index]
    )

    plot_reconstructed_signal(
        time[: index + 1],
        signal[: index + 1],
        reconstructed_t_plus_1_ssm_with_scale,
        data_name=data_name,
        domain_name=recon_domain,
        title=f"SSM (t+1) with scale {scale}",
        paper_ready=PAPER_READY_PLOTS,
    )
    plot_reconstructed_signal(
        time[index - 20 : index + 1],
        signal[index - 20 : index + 1],
        reconstructed_t_plus_1_ssm_with_scale[index - 20 : index + 1],
        data_name=data_name,
        domain_name=recon_domain,
        title=f"SSM (t+1) with scale {scale} (zoomed in)",
        paper_ready=PAPER_READY_PLOTS,
    )

    ########### Without B
    print("========== Reconstruct ck_t_plus_1 using SSM ===========")
    print("========== Without B ===========")
    print("Reconstruct ck_t_plus_1 using SSM WITHOUT B")
    reconstructed_ck_t_plus_1_without_B = np.conjugate(A) @ ck_t
    reconstructed_t_plus_1_ssm_without_B = (
        np.conjugate(Bases_t_plus_1) @ reconstructed_ck_t_plus_1_without_B
    )
    print(
        "reconstructed_t_plus_1_ssm_without_B",
        reconstructed_t_plus_1_ssm_without_B.shape,
    )
    diff_recon_without_B = signal[: index + 1] - reconstructed_t_plus_1_ssm_without_B
    print(
        "difference of reconstructions!(avg) without B: ",
        np.mean(np.abs(diff_recon_without_B)),
    )
    print("Original new data point: ", signal[index])
    print(
        "Reconstructed new data point without B: ",
        reconstructed_t_plus_1_ssm_without_B[index],
    )
    assert (
        reconstructed_t_plus_1_ssm_without_B[-1]
        == reconstructed_t_plus_1_ssm_without_B[index]
    )
    plot_reconstructed_signal(
        time[: index + 1],
        signal[: index + 1],
        reconstructed_t_plus_1_ssm_without_B[: index + 1],
        data_name=data_name,
        domain_name=recon_domain,
        title=f"SSM (t+1) without B*u(t+1) ",
        paper_ready=PAPER_READY_PLOTS,
    )
    plot_reconstructed_signal(
        time[index - 20 : index + 1],
        signal[index - 20 : index + 1],
        reconstructed_t_plus_1_ssm_without_B[index - 20 : index + 1],
        data_name=data_name,
        domain_name=recon_domain,
        title=f"SSM (t+1) without B*u(t+1) (zoomed in)",
        paper_ready=PAPER_READY_PLOTS,
    )

    # Compute Explicit c_t+n

    n = 1
    print(f"Compute Explicit c_t+{n}")
    index_plus_n = index + n
    z_t_plus_n = sigma(time[:index_plus_n], tau)
    weights_t_plus_n = sigma_derivative(time[:index_plus_n], tau)
    ck_t_plus_n, Bases_t_plus_n = compute_coefficients(
        signal[:index_plus_n],
        time[:index_plus_n],
        z_t_plus_n,
        ks,
        weights_t_plus_n,
        basis_type=basis_type,
    )
    reconstructed_t_plus_n = np.conjugate(Bases_t_plus_n) @ ck_t_plus_n

    plot_reconstructed_signal(  # Figure 1
        time[: train_index + n],
        signal[: train_index + n],
        reconstructed_t_plus_n,
        data_name=data_name,
        domain_name=recon_domain,
        title=f"[0,t+{n}] using explicit c_t+{n} (without SSM), ",
        paper_ready=PAPER_READY_PLOTS,
    )
    # plot zvals
    plot_reconstructed_signal(
        z_t_plus_n,
        signal[:index_plus_n],
        reconstructed_t_plus_n,
        data_name=data_name,
        domain_name=recon_domain,
        title=f"[0,t+{n}] using explicit c_t+{n} (without SSM), ",
        paper_ready=PAPER_READY_PLOTS,
    )
    plot_basis_functions(
        x=z_t_plus_n,
        Bases=Bases_t_plus_n,
        ks=ks,
        data_name=data_name,
        domain_name="z",
        basis_type=basis_type,
        paper_ready=PAPER_READY_PLOTS,
    )
    print(
        "difference of Bases between explicit and SSM Bases",
        np.sum(np.abs(Bases_t_plus_n - Bases_t_plus_1_shifted)),
    )

    ########### Compare SSM ck_t_plus_1 with ck_t_plus_n ###########
    print("===== Visually Compare SSM ck_t_plus_1 with Explicit ck_t_plus_n =====")

    plot_signal(
        ks,
        ck_t_plus_n,
        f"Explicit c_t+{n}",
        "n",
    )

    print("===== reconstructed ck_t_plus_1 =====")
    print(
        "Difference (avg) of Transferred ck_t_plus_1 and Explicit ck_t_plus_1: ",
        np.mean(np.abs(reconstructed_ck_t_plus_1 - ck_t_plus_n)),
    )
    plot_signal(
        ks,
        reconstructed_ck_t_plus_1,
        f"SSM-Reconstructed c_t+1",
        "n",
    )
    plot_reconstructed_signal(
        ks,
        reconstructed_ck_t_plus_1,
        ck_t_plus_n,
        f"c_t+{n}",
        "n",
        paper_ready=PAPER_READY_PLOTS,
    )

    print("==== reconstructed ck_t_plus_1 without B =====")
    print(
        "Difference (avg) of Transferred ck_t_plus_1 without B and Explicit ck_t_plus_1: ",
        np.mean(np.abs(reconstructed_ck_t_plus_1_without_B - ck_t_plus_n)),
    )
    plot_signal(
        ks,
        reconstructed_ck_t_plus_1_without_B,
        f"SSM-Reconstructed c_t+1 (without B)",
        "n",
        paper_ready=PAPER_READY_PLOTS,
    )
    plot_reconstructed_signal(
        ks,
        reconstructed_ck_t_plus_1_without_B,
        ck_t_plus_n,
        "" f"c_t+{n} (without B)",
        "n",
        paper_ready=PAPER_READY_PLOTS,
    )

    print(f"==== reconstructed ck_t_plus_1 with B*scale {scale} =====")
    print(
        f"Difference (avg) of Transferred ck_t_plus_1 with B*scale {scale} and Explicit ck_t_plus_1: ",
        np.mean(np.abs(reconstructed_ck_t_plus_1_with_scale - ck_t_plus_n)),
    )
    plot_signal(
        ks,
        reconstructed_ck_t_plus_1_with_scale,
        # f"SSM-Reconstructed c_t+1 (with B*scale {scale})",
        domain_name="n",
        data_name=f"SSM-Reconstructed",
        function_name="c_t+1 (with B*scale)",
        paper_ready=PAPER_READY_PLOTS,
    )
    plot_reconstructed_signal(
        ks,
        reconstructed_ck_t_plus_1_with_scale,
        ck_t_plus_n,
        # f"c_t+{n} (with B*scale {scale})",
        function_name=f"c_t+{n} (with B*scale)",
        domain_name="n",
        paper_ready=PAPER_READY_PLOTS,
    )

    # Compare A * SSM-ck_t_plus_1 with A * ck_t_plus_n
    print("Compare A * SSM-ck_t_plus_1 with A * ck_t_plus_n")
    print("A * SSM-ck_t_plus_1", (A @ reconstructed_ck_t_plus_1).shape)
    print("A * ck_t_plus_n", (A @ ck_t_plus_n).shape)
    plot_reconstructed_signal(
        ks,
        A @ reconstructed_ck_t_plus_1,
        A @ ck_t_plus_n,
        data_name="",
        function_name=f"A * c_t+{n}",
        domain_name="n",
        title=f"[0,t+{n}] ",
        paper_ready=PAPER_READY_PLOTS,
    )


# =========== Compute A_infty and B_ZOH using Gaussian Quadrature =============
print("========== Compute B_ZOH using Gaussian Quadrature ===========")
# %%
from tqdm import tqdm
from scipy.integrate import quad
import time as watch
from scipy.special import roots_legendre, eval_legendre


def phi(n, z):
    """Normalized Legendre polynomial basis function."""
    return eval_legendre(n, z)


# Warp functions for HiPPO-LegS
def sigma(t, s, tau=1):
    return 2 * np.exp((s - t) / tau) - 1


def sigma_inverse(z, t, tau=1):
    return np.log((z + 1) / 2) * tau + t


def get_gauss_quadrature(order, a=0, b=1):
    """Get Gauss-Legendre nodes and weights on interval [a, b]."""
    x, w = roots_legendre(order)  # On [-1, 1]
    z = a + (b - a) * (x + 1) / 2
    w = w * (b - a) / 2
    return z, w


def vectorized_phi(phi_func, n_values, z_values):
    """
    Evaluates a given basis function phi_n(z) for all n in n_values and z in z_values in a vectorized way.

    Args:
    phi_func (callable): Function phi(n, z) returning the basis value for index n at point z.
    n_values (1D array): Array of basis indices [0, 1, ..., N-1].
    z_values (1D array): Array of points to evaluate at [z_0, ..., z_k-1].

    Returns:
    (k, N) array where result[i, j] = phi_func(j, z_values[i]).
    """
    k = len(z_values)
    N = len(n_values)
    phi_vals = np.zeros((k, N))
    for j, n in enumerate(n_values):
        phi_vals[:, j] = phi_func(n, z_values)
    return phi_vals


# Lag operator transition matrices
def lag_transition(N, dt=1.0, quad_order=None):
    if quad_order is None:
        quad_order = (
            2 * N
        )  # Exact for degree < 2N (safe for Legendre products up to degree 2N-2)

    # Direct computation of A_infty via quadrature on [0,1]
    z_nodes_a, z_weights_a = get_gauss_quadrature(quad_order, 0, 1)
    n_vals = np.arange(N)
    eval_n = vectorized_phi(phi, n_vals, z_nodes_a)  # (quad_order, N)
    warped_z = np.exp(-dt) * z_nodes_a
    eval_m = vectorized_phi(phi, n_vals, warped_z)  # (quad_order, N)

    # Fix: Sum over quadrature points to get (N, N)
    A_infty_vec = np.einsum("i,ik,il->kl", z_weights_a, eval_n, eval_m)
    start_time = time.time()
    print(f"\nA_infty computed via vectorized Gauss quadrature (order {quad_order}):")
    print(np.round(A_infty_vec, 3))
    end_time = time.time()
    print(f"Total computation time for A_infty: {end_time - start_time:.2f} seconds")

    # Compute B via quadrature on [exp(-dt), 1]
    z_min = sigma_inverse()
    z_max = 1.0
    z_nodes_b, z_weights_b = get_gauss_quadrature(quad_order, z_min, z_max)
    basis_evals_b = vectorized_phi(phi, n_vals, z_nodes_b)  # (quad_order, N)

    # B = basis_evals_b.T @ z_weights_b (N, quad_order) @ (quad_order,) -> (N,)
    B_zoh_vec = basis_evals_b.T @ z_weights_b

    start_time = time.time()
    print(f"\nB computed via vectorized Gauss quadrature:")
    print(np.round(B_zoh_vec, 3))
    end_time = time.time()
    print(f"Total computation time for B: {end_time - start_time:.2f} seconds")

    return A_infty_vec, B_zoh_vec


A_infty, B_zoh = lag_transition(len(ks), dt=1.0, quad_order=200)

# %%
print(f"\nTotal computation time for B: {end_time - start_time:.2f} seconds")
print("Computed B_t:\n", np.round(B_zoh, 3))

print("========== Reconstruct ck_t_plus_1 using SSM ===========")
print("========== With B_ZOH ===========")

print("Reconstruct ck_t_plus_1 using SSM WITH B_ZOH")

reconstructed_ck_t_plus_1_zoh = np.conjugate(A) @ ck_t + signal[index] * B_zoh
reconstructed_t_plus_1_ssm_zoh = (
    np.conjugate(Bases_t_plus_1) @ reconstructed_ck_t_plus_1_zoh
)
print("reconstructed signal+plus_1 shape", reconstructed_t_plus_1_ssm_zoh.shape)
diff_recon = signal[: index + 1] - reconstructed_t_plus_1_ssm_zoh
print("difference of reconstructions!(avg): ", np.mean(np.abs(diff_recon)))
print("Original new data point: ", signal[index])
print("Reconstructed new data point: ", reconstructed_t_plus_1_ssm_zoh[index])
assert reconstructed_t_plus_1_ssm_zoh[-1] == reconstructed_t_plus_1_ssm_zoh[index]

plot_reconstructed_signal(
    time[: index + 1],
    signal[: index + 1],
    reconstructed_t_plus_1_ssm_zoh,
    data_name=data_name,
    domain_name=recon_domain,
    title=f"SSM (t+1) with ZOH",
    paper_ready=PAPER_READY_PLOTS,
)
plot_reconstructed_signal(
    time[index - 20 : index + 1],
    signal[index - 20 : index + 1],
    reconstructed_t_plus_1_ssm_zoh[index - 20 : index + 1],
    data_name=data_name,
    domain_name=recon_domain,
    title=f"SSM (t+1) with ZOH (zoomed in)",
    paper_ready=PAPER_READY_PLOTS,
)

# %%
