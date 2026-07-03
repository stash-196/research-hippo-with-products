# %%
import numpy as np
import torch
from scipy.linalg import expm
from lag_op_hippo.hippo_by_original_authors import (
    HiPPO,
)
from lag_op_hippo.visualizers import plot_reconstructed_signal
from lag_op_hippo.lag_hippo import (
    sigma,
    sigma_deriv,
    sigma_inv,
    sigma_inv_deriv,
    phi,
    phi_deriv,
    compute_A_infty,
    compute_A_gen,
    LagHiPPO,
    lorenzsignal,
)

# Set seed for reproducibility
np.random.seed(0)

# Global settings
PAPER_READY_PLOTS = True
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
T = 10.0


def matrix_diff(A, B, epsilon=1e-8):
    A = A.copy()
    B = B.copy()
    return np.linalg.norm(A - B, ord="fro") / np.linalg.norm(A, ord="fro")


def run_matrix_comparisons():
    # A_infty and A_gen diffs over dt
    diff_vals_A_infty = []
    N = 64
    dts = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    for dt in dts:
        A_infty = compute_A_infty(N, phi, sigma, sigma_inv, dt=dt, verbose=False)
        A_gen = compute_A_gen(N, phi, phi_deriv, dt=dt, verbose=False)
        A_gen_infty = expm(dt * A_gen)
        mat_diff_A_infty = matrix_diff(A_gen_infty, A_infty)
        diff_vals_A_infty.append(mat_diff_A_infty)

    print("Matrix Diffs between A_infty and A_gen_infty over dt:\n")
    print("dt: diff")
    for dt, val in zip(dts, diff_vals_A_infty):
        formatted_val = np.format_float_scientific(
            val, precision=2, unique=False, trim="."
        )
        print(f"{dt}: {formatted_val:>12}")

    # A_HiPPO vs A_gen over N
    dt = 1e-3
    Ns = [10, 30, 50, 70]
    mat_diffs_A_gen_hippo = []
    for N in Ns:
        hippo_lti = HiPPO(method="legs", N=N, dt=dt, T=T)
        A_HiPPO = hippo_lti.A
        A_gen = compute_A_gen(N, phi, phi_deriv, dt=dt, verbose=False)
        A_gen_HiPPO = -(A_gen + np.eye(N)).T
        mat_diff_A_HiPPO_vs = matrix_diff(A_HiPPO, A_gen_HiPPO)
        mat_diffs_A_gen_hippo.append(mat_diff_A_HiPPO_vs)

    print("Matrix Diffs between A_gen and HiPPO-LegS dA over N:\n")
    print("N: diff")
    for N, val in zip(Ns, mat_diffs_A_gen_hippo):
        formatted_val = np.format_float_scientific(
            val, precision=2, unique=False, trim="."
        )
        print(f"{N}: {formatted_val:>12}")

    # A_infty_corrected and A_HiPPO over dt
    diff_vals_A_infty = []
    N = 64
    dts = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    mat_diffs_A_infty_hippo = []
    for dt in dts:
        A_infty = compute_A_infty(N, phi, sigma, sigma_inv, dt=dt, verbose=False)
        A_infty_corrected = np.linalg.inv(A_infty.T) * np.exp(-dt)
        hippo_lti = HiPPO(method="legs", N=N, dt=dt, T=T)
        dA = hippo_lti.dA.numpy()
        mat_diff_A_infty_hippo = matrix_diff(dA, A_infty_corrected)
        mat_diffs_A_infty_hippo.append(mat_diff_A_infty_hippo)

    print("Matrix Diffs between A_infty_corrected and HiPPO-LegS dA over dt:\n")
    print("dt: diff")
    for dt, val in zip(dts, mat_diffs_A_infty_hippo):
        formatted_val = np.format_float_scientific(
            val, precision=2, unique=False, trim="."
        )
        print(f"{dt}: {formatted_val:>12}")


def run_basis_visualization():
    N = 64
    dt = 1e-3
    T = 3.0
    hippo_lag = LagHiPPO(
        N=N,
        phi_func=phi,
        phi_deriv_func=phi_deriv,
        sigma_func=sigma,
        sigma_deriv_func=sigma_deriv,
        sigma_inv_func=sigma_inv,
        sigma_inv_deriv_func=sigma_inv_deriv,
        dt=dt,
        T=T,
        derive_A_from="infty",
    )
    example_n = N - 1
    coeff_one_hot_at_t = torch.zeros(N)
    coeff_one_hot_at_t[example_n - 1] = 1.0

    A_infty = torch.tensor(
        compute_A_infty(N, phi, sigma, sigma_inv, dt=dt, verbose=False),
        dtype=torch.float,
    )

    A_backward = np.exp(dt) * A_infty.numpy()

    coeff_shifted_to_t_minus_1_from_t = (
        torch.tensor(A_backward, dtype=torch.float) @ coeff_one_hot_at_t
    )
    reconstructed_basis_t_shifted_to_t_minus_1 = (
        hippo_lag.eval_matrix @ coeff_shifted_to_t_minus_1_from_t
    )
    reconstructed_basis_t_shifted_to_t_minus_1[0] = np.nan  # Fix numerical issue at t=0

    zoom = 20

    hippo_lag_t_plus_1 = LagHiPPO(
        N=N,
        phi_func=phi,
        phi_deriv_func=phi_deriv,
        sigma_func=sigma,
        sigma_deriv_func=sigma_deriv,
        sigma_inv_func=sigma_inv,
        sigma_inv_deriv_func=sigma_inv_deriv,
        dt=dt,
        T=T + dt,
        derive_A_from="infty",
    )

    A_forward = torch.linalg.inv(A_infty.T) * np.exp(-dt)

    reconstructed_basis_t_in_t_plus_1 = torch.cat(
        (torch.tensor([np.nan]), hippo_lag.eval_matrix @ coeff_one_hot_at_t)
    ).flip(-1)

    coeff_shifted_to_t_plus_1_from_t = A_forward @ coeff_one_hot_at_t
    reconstructed_basis_t_shifted_to_t_plus_1 = hippo_lag_t_plus_1.reconstruct(
        coeff_shifted_to_t_plus_1_from_t
    )

    vals_t_plus_1 = torch.arange(0.0, T + dt, dt)

    reconstructed_basis_t_shifted_to_t_minus_1_in_t_plus_1 = torch.cat(
        (torch.tensor([np.nan]), reconstructed_basis_t_shifted_to_t_minus_1)
    ).flip(-1)
    reconstructed_list = [
        reconstructed_basis_t_shifted_to_t_plus_1[-zoom:],
        reconstructed_basis_t_shifted_to_t_minus_1_in_t_plus_1[-zoom:],
    ]

    reconstruction_labels = ["Forward-Lagged", "Backward-Lagged"]
    plot_reconstructed_signal(
        vals_t_plus_1[-zoom:],
        reconstructed_basis_t_in_t_plus_1[-zoom:],
        reconstructed_list,
        data_name=f"Basis",
        function_name=rf"$\phi_{{{example_n}}}$",
        domain_name="s",
        title=f"[0,t] ",
        paper_ready=PAPER_READY_PLOTS,
        recon_labels_list=reconstruction_labels,
    )


def run_reconstruction():
    T = 10.0
    N = 64
    freq = 3.0
    dt = 1e-3
    vals = np.arange(0.0, T, dt)
    u = lorenzsignal(T, dt, initial_state=[1.0, 1.0, 8.0])[:, 0]
    u = u / np.std(u)
    u = torch.tensor(u, dtype=torch.float)

    hippo_lti = HiPPO(method="legs", N=N, dt=dt, T=T)
    u_hippo_lti = hippo_lti.reconstruct(hippo_lti(u))[-1].numpy()

    hippo_lag_zoh = LagHiPPO(
        N=N,
        phi_func=phi,
        phi_deriv_func=phi_deriv,
        sigma_func=sigma,
        sigma_deriv_func=sigma_deriv,
        sigma_inv_func=sigma_inv,
        sigma_inv_deriv_func=sigma_inv_deriv,
        dt=dt,
        T=T,
        derive_A_from="infty",
        input_hold="zoh",
    )

    u_hippo_lag_zoh = hippo_lag_zoh.reconstruct(hippo_lag_zoh(u))[-1].numpy()

    diff_u_hippo_lag_zoh = np.linalg.norm(
        u_hippo_lti - u_hippo_lag_zoh
    ) / np.linalg.norm(u_hippo_lti)
    print(
        f"Relative difference between u_hippo_lti and u_hippo_lag_zoh: {diff_u_hippo_lag_zoh:.6f}"
    )

    recon_list = [
        u_hippo_lti,
        u_hippo_lag_zoh,
        hippo_lti.measure[::-1] - 3.0,
    ]
    reconstruction_labels = [
        "HiPPO-LegS",
        "LagOp HiPPO-ZOH",
        "Exponential Measure",
    ]
    recon_linestyle_list = [
        "-",
        "-.",
        "-",
    ]

    plot_reconstructed_signal(
        vals,
        u,
        recon_list,
        data_name="",
        domain_name="s",
        title=f"Input Frequency = {freq} Hz",
        paper_ready=PAPER_READY_PLOTS,
        recon_labels_list=reconstruction_labels,
        recon_linestyle_list=recon_linestyle_list,
        recon_color_list=None,
        function_name="u",
        legend_loc="upper left",
        xlim=(vals[0] - 1, vals[-1]),
        ylim=(-3.5, 3.5),
    )
    recon_mse_vs_hippo = np.mean((u_hippo_lti - u_hippo_lag_zoh) ** 2)
    print(
        "MSE between HiPPO-LegS and LagHiPPO-ZOH reconstructions:", recon_mse_vs_hippo
    )


if __name__ == "__main__":
    run_matrix_comparisons()
    run_basis_visualization()
    run_reconstruction()
# %%
