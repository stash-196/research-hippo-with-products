# %%
import time
import numpy as np
import torch
from scipy.linalg import expm
from lag_op_hippo.hippo_by_original_authors import (
    HiPPO,
)
from scipy.interpolate import interp1d
from lag_op_hippo.visualizers import plot_basis_functions, plot_reconstructed_signal
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
    get_gauss_quadrature,
    vectorized_basis,
)


# Parameters (tune these)
ALPHA = 0.2  # Decay rate (close to 1 for longer memory)
L_FOURIER = 1  # Window length (set to k / target_freq; e.g., 0.1 for ~10 Hz with k=1)
K_FOURIER = 1  # Fourier mode (higher k for higher freq per window)
EPS = 1e-8  # Numerical stability for bounds/log


def f_tau(tau):
    """f(τ) for τ >= 0 (z = f(τ))."""
    tau = np.asarray(tau)
    tau = np.clip(tau, 0, np.inf)
    n = np.floor(tau / L_FOURIER) + 1
    n = n.astype(int)
    alpha_n = ALPHA**n
    h_n = (1 - ALPHA) * (ALPHA ** (n - 1))
    frac = n - tau / L_FOURIER
    z = alpha_n + h_n * frac
    return np.clip(z, 0, 1)


def sigma_fourier(t, s):
    tau = t - s
    return f_tau(tau)


def sigma_deriv_fourier(t, s):
    tau = t - s
    n = np.floor(tau / L_FOURIER) + 1
    n = n.astype(int)
    h_n = (1 - ALPHA) * (ALPHA ** (n - 1))
    return h_n / L_FOURIER


def sigma_inv_fourier(t, z):
    z = np.asarray(z)
    z = np.clip(z, EPS, 1)
    p = np.log(z) / np.log(ALPHA)
    n = np.floor(p) + 1
    n = n.astype(int)
    alpha_n = ALPHA**n
    h_n = (1 - ALPHA) * (ALPHA ** (n - 1))
    tau = L_FOURIER * (n - (z - alpha_n) / h_n)
    s = t - tau
    return s


def sigma_inv_deriv_fourier(t, z):
    z = np.asarray(z)
    z = np.clip(z, EPS, 1)
    p = np.log(z) / np.log(ALPHA)
    n = np.floor(p) + 1
    n = n.astype(int)
    h_n = (1 - ALPHA) * (ALPHA ** (n - 1))
    return L_FOURIER / h_n


# def psi_fourier(n, s):
#     """Time basis ψ_n(s): cos for even n, sin for odd n, with amplitude for orthonormality."""
#     k = n + 1
#     m = k // 2
#     # j = m + 1  # Window index
#     # h_j = (1 - ALPHA) * (ALPHA ** (j - 1))
#     arg = 2 * np.pi * m * s / L_FOURIER
#     amp_const = np.sqrt(1.0)  # For n=0
#     amp_trig = np.sqrt(2.0)
#     if k % 2 == 1:  # cos
#         val = amp_trig * np.cos(arg) if k > 1 else amp_const * np.cos(arg)
#     else:  # sin
#         val = amp_trig * np.sin(arg)
#     return val


def psi_fourier(n, s):
    k = n + 1
    m = k // 2
    tau = np.asarray(s)
    n_win = np.floor(tau / L_FOURIER) + 1
    n_win = n_win.astype(int)
    h_n = (1 - ALPHA) * (ALPHA ** (n_win - 1))
    amp = np.sqrt(1.0 / h_n)  # ← THIS WAS THE MISSING PIECE
    arg = 2 * np.pi * m * s / L_FOURIER
    if k % 2 == 1:
        val = amp * np.cos(arg)
    else:
        val = amp * np.sin(arg)
    return val


def phi_fourier(n, z):
    """Fourier basis φ_n(z): using time basis ψ_n(s) and sigma_inv."""
    s = sigma_inv_fourier(0, z)  # t=0 for basis functions
    return psi_fourier(n, s)


def psi_deriv_fourier(n, s):
    """Derivative of ψ_n(s)."""
    m = n // 2
    arg = 2 * np.pi * K_FOURIER * s / L_FOURIER
    darg_ds = 2 * np.pi * K_FOURIER / L_FOURIER
    # amp = np.sqrt(2 / h_j)
    # amp = 1.0  # Unnormalized
    amp = np.sqrt(1 / L_FOURIER)
    if n % 2 == 0:  # d(cos)/ds = -sin * darg_ds
        return amp * (-np.sin(arg) * darg_ds)
    else:  # d(sin)/ds = cos * darg_ds
        return amp * (np.cos(arg) * darg_ds)


def phi_deriv_fourier(n, z):
    """Derivative of φ_n(z) using chain rule."""
    s = sigma_inv_fourier(0, z)
    ds_dz = sigma_inv_deriv_fourier(0, z)
    return psi_deriv_fourier(n, s) * ds_dz


N_FOURIER = 5  # Must be even (sin+cos per window); M = N//2 =16 windows
PAPER_READY_PLOTS = True
# Test signal (high freq ~10 Hz to match L=0.1, k=1)
vals = np.arange(0.0, 10.0, 0.001)
sigmas_as_basis = np.array(
    [sigma_deriv_fourier(max(vals), vals), sigma_fourier(max(vals), vals)]
).T
plot_basis_functions(  # Figure 4a
    x=sigma_fourier(10, vals),
    Bases=sigmas_as_basis,
    ks=range(N_FOURIER),
    data_name="sigmas",
    domain_name="z",
    basis_type="frequency-preserving",
    title=f"sigma [0,t] ",
    paper_ready=PAPER_READY_PLOTS,
)

plot_basis_functions(  # Figure 4a
    x=vals,
    Bases=sigmas_as_basis,
    ks=range(N_FOURIER),
    data_name="sigmas",
    domain_name="z",
    basis_type="frequency-preserving",
    title=f"sigma [0,1] ",
    paper_ready=PAPER_READY_PLOTS,
)

# %%
# In lag_hippo.py or your test script
N_FOURIER = 5  # Must be even (sin+cos per window); M = N//2 =16 windows
PAPER_READY_PLOTS = True

model_fourier = LagHiPPO(
    N=N_FOURIER,
    phi_func=phi_fourier,
    phi_deriv_func=phi_deriv_fourier,
    sigma_func=sigma_fourier,
    sigma_deriv_func=sigma_deriv_fourier,
    sigma_inv_func=sigma_inv_fourier,
    sigma_inv_deriv_func=sigma_inv_deriv_fourier,
    dt=0.001,  # Smaller dt for high freq
    T=10.0,
    derive_A_from="infty",  # Or "infty"; test both for stability
    input_hold="zoh",  # Or "zoh"
    quad_order=200 * N_FOURIER,  # Higher order for accuracy
)

# Test signal (high freq ~10 Hz to match L=0.1, k=1)
vals = np.arange(0.0, 10.0, 0.001)
u = np.sin(2 * np.pi * 2 * vals)  # Pure sine at 10 Hz
u_torch = torch.tensor(u, dtype=torch.float)

c_fourier = model_fourier.forward(u_torch)
recon_fourier = model_fourier.reconstruct(c_fourier[-1]).numpy()

mse_fourier = np.mean((u - recon_fourier) ** 2)
print(f"MSE Fourier: {mse_fourier}")

# Compare to standard (exponential + Legendre)
model_standard = LagHiPPO(
    N=N_FOURIER,
    phi_func=phi,
    phi_deriv_func=phi_deriv,
    sigma_func=sigma,
    sigma_deriv_func=sigma_deriv,
    sigma_inv_func=sigma_inv,
    sigma_inv_deriv_func=sigma_inv_deriv,
    derive_A_from="infty",  # Or "gen"
    input_hold="zoh",  # Or "zoh"
    dt=0.001,
    T=10.0,
)


# %%
plot_basis_functions(  # Figure 4a
    x=vals,
    Bases=model_standard.eval_matrix.numpy()[::-1],
    ks=range(N_FOURIER),
    data_name="Basis Functions",
    domain_name="s",
    basis_type="frequency-preserving",
    title=f"Psi_[0,t] ",
    paper_ready=PAPER_READY_PLOTS,
)
plot_basis_functions(  # Figure 4a
    x=sigma(vals, 0),
    Bases=model_standard.eval_matrix,
    ks=range(N_FOURIER),
    data_name="Basis Functions",
    domain_name="z",
    basis_type="frequency-preserving",
    title=f"Phi_[0,t] ",
    paper_ready=PAPER_READY_PLOTS,
)
plot_basis_functions(  # Figure 4a
    x=vals,
    Bases=model_fourier.eval_matrix.numpy()[::-1],
    ks=range(N_FOURIER),
    data_name="Basis Functions",
    domain_name="s",
    basis_type="frequency-preserving",
    title=f"Psi_[0,t] ",
    paper_ready=PAPER_READY_PLOTS,
)
plot_basis_functions(  # Figure 4a
    x=sigma_fourier(vals, 0),
    Bases=model_fourier.eval_matrix,
    ks=range(N_FOURIER),
    data_name="Basis Functions",
    domain_name="z",
    basis_type="frequency-preserving",
    title=f"Phi_[0,t] ",
    paper_ready=PAPER_READY_PLOTS,
)

recon_list = [
    sigma_fourier(max(vals), vals),
    model_standard._compute_measure()[::-1] - 3.0,
    model_fourier._compute_measure()[::-1] - 3.0,
]
reconstruction_labels = [
    "Freq-Preserving Time Warp",
    "Standard Measure",
    "Freq-Preserving Measure",
]
recon_linestyle_list = [
    "-",
    "--",
    "-",
]
freq = 0.001
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

# %%
c_standard = model_standard.forward(u_torch)
recon_standard = model_standard.reconstruct(c_standard[-1]).numpy()
mse_standard = np.mean((u - recon_standard) ** 2)
print(f"MSE Standard: {mse_standard}")

diff_u_hippo_lag_zoh = np.linalg.norm(recon_fourier - recon_standard) / np.linalg.norm(
    recon_standard
)
print(
    f"Relative difference between u_hippo_lti and u_hippo_lag_zoh: {diff_u_hippo_lag_zoh:.6f}"
)

recon_list = [
    recon_standard,
    recon_fourier,
    model_standard._compute_measure()[::-1] - 3.0,
    model_fourier._compute_measure()[::-1] - 3.0,
]
reconstruction_labels = [
    "LagOp HiPPO-Standard",
    "LagOp HiPPO-Freq-Preserving",
    "Standard Measure",
    "Freq-Preserving Measure",
]
recon_linestyle_list = [
    "-",
    "-.",
    "--",
    "-",
]
freq = 0.001
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
recon_mse_vs_hippo = np.mean((recon_standard - recon_fourier) ** 2)
print(
    "MSE between LagOp HiPPO-Standard and LagOp HiPPO-Freq-Preserving reconstructions:",
    recon_mse_vs_hippo,
)


# %%
# New modular inner product function
def compute_inner_product(phi_func, u_func, sigma_inv_func, N, t=0.0, quad_order=None):
    """
    Compute coefficients c_n = <phi_n, u ∘ sigma_inv^{-1}> = ∫_0^1 phi_n(z) u(sigma_inv_t(z)) dz
    Using Gauss-Legendre quadrature. This projects the signal u onto the basis.

    Args:
        phi_func: Basis function phi(n, z)
        u_func: Signal function u(s) -> value (callable; use interp1d for discrete u)
        sigma_inv_func: Inverse warp sigma_inv(t, z) -> s
        N: Number of basis functions (state dim)
        t: Current time (default 0.0; assumes history s <= t)
        quad_order: Quadrature order (default 2*N for accuracy)

    Returns:
        c: np.array (N,) coefficients
    """
    if quad_order is None:
        quad_order = max(2 * N, 4)  # Minimal fallback, but 2*N is usually sufficient
    z_nodes, weights = get_gauss_quadrature(quad_order, 0, 1)
    s_vals = sigma_inv_func(t, z_nodes)
    phi_vals = vectorized_basis(phi_func, np.arange(N), z_nodes)  # (quad_order, N)
    u_vals = u_func(s_vals)
    # c_j = sum_i weights_i * phi_j(z_i) * u(s_i)
    c = np.einsum("i, ij, i -> j", weights, phi_vals, u_vals)
    return c


# New modular reconstruction function
def reconstruct_signal(s_vals, c, phi_func, sigma_func, t=0.0):
    """
    Reconstruct u_recon(s) = sum_n c_n psi_{t,n}(s) = sum_n c_n phi_n(sigma_t(s))

    Args:
        s_vals: np.array of s points to evaluate reconstruction at
        c: np.array (N,) coefficients
        phi_func: Basis function phi(n, z)
        sigma_func: Warp sigma(t, s) -> z
        t: Current time (default 0.0)

    Returns:
        u_recon: np.array (len(s_vals),) reconstructed signal
    """
    z_vals = sigma_func(t, s_vals)
    phi_vals = vectorized_basis(phi_func, np.arange(len(c)), z_vals)  # (len(s_vals), N)
    u_recon = phi_vals @ c
    return u_recon


# Test function
def test_projection_reconstruction(
    u_func, s_vals, phi_func, sigma_func, sigma_inv_func, N, t=0.0, quad_order=None
):
    """
    Test direct projection and reconstruction on a signal (non-recurrent; for validation).

    Args:
        u_vals: np.array discrete signal values (assumed over s_vals; u=0 outside)
        s_vals: np.array corresponding s points (e.g., np.linspace(s_min, t, num_points))
        phi_func: Basis phi(n, z)
        sigma_func: Warp sigma(t, s) -> z
        sigma_inv_func: Inverse sigma_inv(t, z) -> s
        N: State dim
        t: Time (default 0.0; project/reconstruct at this t)
        quad_order: For inner product (default 2*N)

    Returns:
        c_t: Coefficients (N,)
        u_recon: Reconstructed at s_vals
        mse: Mean squared error vs original u at s_vals

    Notes:
        - Assumes u=0 for s not in s_vals (e.g., distant history).
        - Uses linear interpolation for u_func; for better accuracy on infinite history,
          extend s_vals to cover effective support (where omega_t > epsilon).
    """

    u_vals = u_func(s_vals)
    start_time = time.time()
    c_t = compute_inner_product(
        phi_func, u_func, sigma_inv_func, N, t=t, quad_order=quad_order
    )
    u_recon = reconstruct_signal(s_vals, c_t, phi_func, sigma_func, t=t)
    mse = np.mean((u_recon - u_vals) ** 2)
    end_time = time.time()
    print(
        f"Projection-reconstruction test completed in {end_time - start_time:.4f} seconds."
    )
    print(f"Mean Squared Error (MSE): {mse:.6e}")
    return c_t, u_recon, mse


# Create callable u_func with interpolation (u=0 outside s_vals bounds)
T = 1.0
dt = 0.001
vals = np.arange(0.0, T, dt)
u = lorenzsignal(T, dt, initial_state=[1.0, 1.0, 8.0])[:, 0]
u = (u - np.mean(u)) / np.std(u)  # Normalize
s_vals = np.linspace(-0.0, T, int(T / dt) + 1)  # Extend left for history (u=0 there)
u_extended = np.zeros_like(s_vals)
mask = (s_vals >= 0) & (s_vals <= T)
u_extended[mask] = np.interp(s_vals[mask], vals, u)  # Align to original vals
u_func = interp1d(s_vals, u_extended, kind="linear", bounds_error=False, fill_value=0.0)
c_exponential, u_recon_exponential, mse_exponential = test_projection_reconstruction(
    u_func, s_vals, phi, sigma, sigma_inv, N=64, t=T
)
c_fourier, u_recon_fourier, mse_fourier = test_projection_reconstruction(
    u_func, s_vals, phi_fourier, sigma_fourier, sigma_inv_fourier, N=64, t=T
)

plot_reconstructed_signal(
    s_vals,
    u_extended,
    [u_recon_exponential, u_recon_fourier],
    data_name="",
    domain_name="s",
    title=f"TEST",
    paper_ready=PAPER_READY_PLOTS,
    recon_labels_list=["Exponential_Reconstruction", "Fourier_Reconstruction"],
    recon_linestyle_list=["-", "-"],
    recon_color_list=None,
    function_name="u",
    legend_loc="upper left",
    xlim=(s_vals[0] - 1, s_vals[-1]),
    ylim=(-3.5, 3.5),
)


# %%
# Test orthonormality
from scipy.integrate import quad


from scipy.integrate import quad


def test_orthonormality(N, phi, Z=(0, 1)):
    for n in range(N):
        for m in range(N):

            def integrand(z):
                return phi(n, z) * phi(m, z)

            result, err = quad(
                integrand, Z[0], Z[1], epsabs=1e-12, epsrel=1e-12, limit=10000
            )
            # print(f"n={n}, m={m}: integral={result:.3e}, est_err={err:.3e}")  # Debug
            if n == m:
                # assert np.isclose(result, 1, atol=1e-8), f"Failed norm at {n},{m}: {result}"
                pass
            else:
                assert np.isclose(
                    result, 0, atol=1e-8
                ), f"Failed ortho at {n},{m}: {result}"
    print("Orthogonality test passed!")


N_FOURIER_TEST = 5
print("Testing orthogonality for Fourier basis for order of basis N =", N_FOURIER_TEST)


print("Over interval [(1-Sum h_j), (1-Sum h_j-1)]:")
for N in range(N_FOURIER_TEST + 1):
    Z_min = sigma_fourier(10, 10 - (N) * L_FOURIER)
    Z_max = sigma_fourier(10, 10 - (N - 1) * L_FOURIER)
    print(
        f"  N={N} over [{Z_min:.3g}, {Z_max:.3g}]",
        end=":",
    )
    try:
        test_orthonormality(N=N, phi=phi_fourier, Z=(Z_min, Z_max))
    except AssertionError as e:
        print(
            f"Orthogonality test failed for N={N} on interval [{Z_min:.3g}, {Z_max:.3g}]: {e}"
        )
        break

print("Over multiple intervals [(1-Sum h_j), 1]:")
for N in range(2, N_FOURIER_TEST + 1):
    Z_min = sigma_fourier(10, 10 - (N) * L_FOURIER)
    Z_max = 1.0
    print(
        f"  N={N} over [{Z_min:.3g}, {Z_max:.3g}]",
        end=":",
    )
    try:
        test_orthonormality(N=N, phi=phi_fourier, Z=(Z_min, Z_max))
    except AssertionError as e:
        print(
            f"\n    Orthogonality test failed for N={N} on interval [{Z_min:.3g}, {Z_max:.3g}]: {e}"
        )
        break


# %%
def test_orthonormality_gauss(N, phi, Z=(0, 1), quad_order=1000, tolerance=1e-4):
    z_nodes, weights = get_gauss_quadrature(quad_order, Z[0], Z[1])
    phi_vals = vectorized_basis(phi, np.arange(N), z_nodes)  # (quad_order, N)
    inner_mat = np.einsum("i, ik, il -> kl", weights, phi_vals, phi_vals)
    print(f"Inner matrix for N={N} over {Z}:")
    print(np.round(inner_mat, 5))
    for n in range(N):
        for m in range(N):
            if n == m:
                if not np.isclose(
                    inner_mat[n, m], inner_mat[0, 0], atol=1e-4
                ):  # Norms consistent with first
                    print(f"Norm vary at n={n}: {inner_mat[n,n]}")
                    print(f"Norm({n},{m}): {inner_mat[n,m]:.3g}", end=", ")
                pass
            else:
                if not np.isclose(inner_mat[n, m], 0, atol=tolerance):
                    # print(f"    Off-diag at {n},{m}: {inner_mat[n,m]}")
                    print(f"Off({n},{m}): {inner_mat[n,m]:.3g}", end=", ")
    print(f"Test complete (tol={tolerance}).")
    return inner_mat


# %%
N_FOURIER_TEST = 20
quad_order = max(
    2 * N_FOURIER_TEST,
    4 * N_FOURIER_TEST if N_FOURIER_TEST > 20 else 2 * N_FOURIER_TEST,
)
quad_order = 100 * N_FOURIER_TEST

print(
    "Testing orthogonality for Fourier basis for order of basis N =",
    N_FOURIER_TEST,
    f"with quad_order={quad_order}",
)


print("Over interval [(1-Sum h_j), (1-Sum h_j-1)]:")
for N in range(N_FOURIER_TEST + 1):
    Z_min = sigma_fourier(10, 10 - (N) * L_FOURIER)
    Z_max = sigma_fourier(10, 10 - (N - 1) * L_FOURIER)
    print(f"  N={N} over [{Z_min:.3g}, {Z_max:.3g}]", end=":")
    try:
        test_orthonormality_gauss(
            N=N,
            phi=phi_fourier,
            Z=(Z_min, Z_max),
            quad_order=quad_order,
            tolerance=1e-8,
        )
    except AssertionError as e:
        print(
            f"Orthogonality test failed for N={N} on interval [{Z_min:.3g}, {Z_max:.3g}]: {e}"
        )
        break

print("Over multiple intervals [(1-Sum h_j), 1]:")
for N in range(2, N_FOURIER_TEST + 1):
    Z_min = sigma_fourier(10, 10 - (N) * L_FOURIER)
    Z_max = 1.0
    print(
        f"  N={N} over [{Z_min:.3g}, {Z_max:.3g}]",
        end=":",
    )
    try:
        test_orthonormality_gauss(
            N=N,
            phi=phi_fourier,
            Z=(Z_min, Z_max),
            quad_order=quad_order,
            tolerance=1e-4,
        )
    except AssertionError as e:
        print(
            f"\n    Orthogonality test failed for N={N} on interval [{Z_min:.3g}, {Z_max:.3g}]: {e}"
        )
        break
# %%
N_FOURIER_TEST = 5
print("Final comprehensive orthonormality test over [0,1]:")
inner_mat = test_orthonormality_gauss(
    N=N_FOURIER_TEST,
    phi=phi_fourier,
    Z=(0, 1),
    quad_order=quad_order * 20,
    tolerance=1e-5,
)

# %%
