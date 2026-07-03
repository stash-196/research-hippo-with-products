# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import your modules
# Assuming the file structure allows these imports, or place this script in the root
from lag_op_hippo.hippo_by_original_authors import HiPPO
from lag_op_hippo.lag_hippo import (
    LagHiPPO,
    lorenzsignal,
    phi,
    phi_deriv,
    sigma,
    sigma_deriv,
    sigma_inv,
    sigma_inv_deriv,
)
from lag_op_hippo.lag_hippo_split import LagHiPPOSplit
from lag_op_hippo.visualizers import plot_reconstructed_signal


# %%
# --- Define Fast/Slow Warps for the Split Model ---
def sigma_fast(t, s):
    return np.exp(2 * (s - t))


def sigma_deriv_fast(t, s):
    return 2 * np.exp(2 * (s - t))


def sigma_inv_fast(t, z):
    return t + 0.5 * np.log(z)


def sigma_inv_deriv_fast(t, z):
    return 0.5 / z


def sigma_slow(t, s):
    return np.exp(0.5 * (s - t))


def sigma_deriv_slow(t, s):
    return 0.5 * np.exp(0.5 * (s - t))


def sigma_inv_slow(t, z):
    return t + 2 * np.log(z)


def sigma_inv_deriv_slow(t, z):
    return 2 / z


def run_split_reconstruction():
    # 1. Setup Data
    print("Generating Lorenz Signal...")
    T = 10.0
    dt = 1e-3
    N = 64
    vals = np.arange(0.0, T, dt)

    # Generate Lorenz signal (x-axis)
    u_raw = lorenzsignal(T, dt, initial_state=[1.0, 1.0, 8.0])[:, 0]
    u_raw = u_raw / np.std(u_raw)
    u = torch.tensor(u_raw, dtype=torch.float)

    # 2. Run Standard HiPPO (Original)
    print(f"Running HiPPO-LegS (N={N})...")
    hippo_orig = HiPPO(method="legs", N=N, dt=dt, T=T)
    c_orig = hippo_orig(u)
    # Reconstruct at final step
    u_orig = hippo_orig.reconstruct(c_orig)[-1].detach().numpy()

    # 3. Run LagHiPPO (Standard LagOp)
    print(f"Running LagHiPPO (Standard, N={N})...")
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
        derive_A_from="infty",  # Use exact discrete matrix
        input_hold="zoh",
    )
    c_lag = hippo_lag(u)
    u_lag = hippo_lag.reconstruct(c_lag)[-1].detach().numpy()

    # 4. Run LagHiPPOSplit (Multi-Resolution: Fast/Slow)
    # Split N into two blocks of 32
    split_sizes = [N // 2, N // 2]
    print(f"Running LagHiPPOSplit (Fast/Slow, N={split_sizes})...")

    hippo_split = LagHiPPOSplit(
        N=N,
        phi_func=phi,
        phi_deriv_func=phi_deriv,
        sigma_func=[sigma_fast, sigma_slow],
        sigma_deriv_func=[sigma_deriv_fast, sigma_deriv_slow],
        sigma_inv_func=[sigma_inv_fast, sigma_inv_slow],
        sigma_inv_deriv_func=[sigma_inv_deriv_fast, sigma_inv_deriv_slow],
        dt=dt,
        T=T,
        split_sizes=split_sizes,
        derive_A_from="gen",  # Split implementation in provided file uses 'gen' by default construction logic
        input_hold="delta",  # Default for split class provided
    )
    c_split = hippo_split(u)
    u_split = hippo_split.reconstruct(c_split)[-1].detach().numpy()

    # 5. Metrics
    mse_orig_vs_lag = np.mean((u_orig - u_lag) ** 2)
    mse_orig_vs_split = np.mean((u_orig - u_split) ** 2)
    mse_input_vs_split = np.mean((u_raw - u_split) ** 2)
    mse_input_vs_orig = np.mean((u_raw - u_orig) ** 2)

    print("\n--- Results ---")
    print(f"MSE (HiPPO vs LagHiPPO): {mse_orig_vs_lag:.2e} (Should be ~0)")
    print(f"MSE (Input vs HiPPO):    {mse_input_vs_orig:.4f}")
    print(f"MSE (Input vs Split):    {mse_input_vs_split:.4f}")

    # 6. Visualization
    recon_list = [u_orig, u_lag, u_split]
    labels = [
        "HiPPO-LegS (Standard)",
        "LagHiPPO (Standard)",
        "LagHiPPOSplit (Fast/Slow)",
    ]
    linestyles = ["--", ":", "-."]
    colors = ["black", "orange", "red"]

    plot_reconstructed_signal(
        vals,
        u_raw,
        recon_list,
        data_name="Lorenz-X",
        domain_name="s",
        title="Multi-Resolution Memory Decomposition",
        paper_ready=False,
        recon_labels_list=labels,
        recon_linestyle_list=linestyles,
        recon_color_list=colors,
        ylim=(-3, 3),
        xlim=(0, 10),
    )


if __name__ == "__main__":
    run_split_reconstruction()
# %%
