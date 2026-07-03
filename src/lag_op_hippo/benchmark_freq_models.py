"""Benchmark runner for standard vs frequency-preserving LagHiPPO models."""

import time
import numpy as np
import torch

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
from lag_op_hippo.lag_hippo_freq import make_frequency_preserving_model


def _run_model(model, u):
    start = time.perf_counter()
    coeffs = model.forward(u)
    recon = model.reconstruct(coeffs[-1]).numpy()
    elapsed_s = time.perf_counter() - start
    return recon, elapsed_s


def run_benchmark(
    N=64,
    T=10.0,
    dt=1e-3,
    alpha=0.2,
    window_length=1.0,
    seed=0,
):
    np.random.seed(seed)

    vals = np.arange(0.0, T, dt)
    u_np = lorenzsignal(T, dt, initial_state=[1.0, 1.0, 8.0])[:, 0]
    u_np = (u_np - np.mean(u_np)) / np.std(u_np)
    u = torch.tensor(u_np, dtype=torch.float)

    standard = LagHiPPO(
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

    freq = make_frequency_preserving_model(
        N=N,
        dt=dt,
        T=T,
        alpha=alpha,
        window_length=window_length,
        derive_A_from="infty",
        input_hold="zoh",
        quad_order=200 * N,
    )

    recon_standard, t_standard = _run_model(standard, u)
    recon_freq, t_freq = _run_model(freq, u)

    mse_standard = np.mean((u_np - recon_standard) ** 2)
    mse_freq = np.mean((u_np - recon_freq) ** 2)
    rel_diff = np.linalg.norm(recon_standard - recon_freq) / np.linalg.norm(
        recon_standard
    )

    print("Benchmark results")
    print(f"N={N}, T={T}, dt={dt}, alpha={alpha}, window_length={window_length}")
    print(f"Standard runtime (s):          {t_standard:.4f}")
    print(f"Frequency runtime (s):         {t_freq:.4f}")
    print(f"Standard MSE vs input:         {mse_standard:.6e}")
    print(f"Frequency MSE vs input:        {mse_freq:.6e}")
    print(f"Relative diff (std vs freq):   {rel_diff:.6e}")

    return {
        "runtime_standard_s": float(t_standard),
        "runtime_frequency_s": float(t_freq),
        "mse_standard": float(mse_standard),
        "mse_frequency": float(mse_freq),
        "relative_difference": float(rel_diff),
    }


if __name__ == "__main__":
    run_benchmark()
