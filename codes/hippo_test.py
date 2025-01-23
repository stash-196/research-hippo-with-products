# %%
import os
import sys
import pickle
import numpy as np
import finufft
import matplotlib.pyplot as plt

# load the pickle file
file_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.join(file_dir, "..", ".."))
data_dir = os.path.join(root_dir, "data")


# %%
# def zeta(t, s, tau=1):
#     return 2 * np.pi * np.exp((s - t) / tau) - np.pi
def zeta(t, s, tau=1):
    return 2 * np.exp((s - t) / tau) - 1


# def zeta_derivative(t, s, tau=1):
#     return 2 * np.pi * np.exp((s - t) / tau) / tau
def zeta_derivative(t, s, tau=1):
    return 2 * np.exp((s - t) / tau) / tau


def inner_product(f, g, w):
    return np.sum(f * np.conj(g) * w)


# %%
whitenoise_saved_data_path = os.path.join(data_dir, "whitenoise_samples.pkl")

with open(whitenoise_saved_data_path, "rb") as f:
    whitenoise_data = pickle.load(f)

print(whitenoise_data)

# Example usage of zeta with half data:
whitenoise_half_point = len(whitenoise_data["time"]) // 2
whitenoise_train_time = whitenoise_data["time"][:whitenoise_half_point]  # s_i
whitenoise_train_data = whitenoise_data["data"][
    :whitenoise_half_point
]  # f_i in the original domain
# Pick a reference time for the transformation
t_ref = whitenoise_data["time"][whitenoise_half_point]
tau = 5

whitenoise_t_ref = whitenoise_data["time"][whitenoise_half_point]
whitenoise_z_vals = zeta(whitenoise_t_ref, whitenoise_train_time, tau)
whitenoise_f_vals = np.asarray(whitenoise_train_data, dtype=np.complex128)

# Define the Fourier basis range
K = 10000
ks = np.arange(-K, K + 1)

# Phi = np.array([np.exp(-1j * k * whitenoise_z_vals) for k in ks]).T

# Psi = np.array([np.exp(-1j * k * whitenoise_train_time) for k in ks]).T
Psi = np.array([np.exp(-1j * np.pi * k * whitenoise_train_time) for k in ks]).T

w_vec = zeta_derivative(whitenoise_t_ref, whitenoise_train_time, tau)


ck = (whitenoise_f_vals * w_vec) @ (Psi)
# Normalize by the number of points
ck /= len(ck)
# Reconstruct F(z) using the Fourier coefficients
whitenoise_f_vals_hat = np.sum(
    [
        #   ck[m] * np.exp(1j * ks[m] * whitenoise_train_time)
        ck[m] * np.exp(1j * np.pi * ks[m] * whitenoise_train_time)
        for m in range(len(ks))
    ],
    axis=0,
)
# Compare original and reconstructed functions
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(whitenoise_train_time, whitenoise_f_vals, label="Original f(s)")

plt.plot(
    whitenoise_train_time,
    np.real(whitenoise_f_vals_hat),
    label="Reconstructed f(s)",
    linestyle="--",
)
plt.legend()
plt.xlabel("s")
plt.ylabel("f(s)")
plt.title("Original vs Reconstructed Whitenoise Function (when w(s) = 1)")
plt.grid(True)
plt.show()
# %%

# Convert to z domain
zvals = zeta(whitenoise_t_ref, whitenoise_train_time)  # shape ~ (half_point,)


# Now we have a new function f(zvals) = fz
fz = np.asarray(whitenoise_train_data, dtype=np.complex128)  # Ensure complex type

# (Check shapes)
print("zvals shape:", zvals.shape)
print("fz shape:", fz.shape)


# %%
# Plot the function in the s domain
plt.figure(figsize=(10, 6))
plt.plot(whitenoise_train_time, np.real(whitenoise_train_data), label="Re[f(s)]")
plt.plot(
    whitenoise_train_time,
    np.imag(whitenoise_train_data),
    label="Im[f(s)]",
    linestyle="--",
)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Whitenoise Function in s domain")
plt.grid(True)
plt.legend()
plt.show()

# Plot the function in the z domain
plt.figure(figsize=(10, 6))
plt.plot(zvals, np.real(fz), label="Re[F(z)]")
plt.plot(zvals, np.imag(fz), label="Im[F(z)]", linestyle="--")
plt.xlabel("Time (z)")
plt.ylabel("Amplitude")
plt.title("Whitenoise Function in z domain")
plt.grid(True)
plt.legend()
plt.show()


# %%

lorenz63_saved_data_path = os.path.join(data_dir, "lorenz63_samples.pkl")

with open(lorenz63_saved_data_path, "rb") as f:
    lorenz63_data = pickle.load(f)

# Example usage of zeta with half data:
lorenz63_half_point = len(lorenz63_data["time"]) // 2
lorenz63_train_timestamps = lorenz63_data["time"][:lorenz63_half_point]  # s_i
lorenz63_train_data = np.asarray(lorenz63_data["data"])[
    :lorenz63_half_point, 0
]  # f_i in the original domain
# Pick a reference time for the transformation
lorenz63_t_ref = lorenz63_data["time"][lorenz63_half_point]

lorenz63_z_vals = zeta(lorenz63_t_ref, lorenz63_train_timestamps, tau)
lorenz63_f_vals = np.asarray(lorenz63_train_data, dtype=np.complex128)

# Define the Fourier basis range
K = 1000
ks = np.arange(-K, K + 1)
tau = 5

# Phi = np.array([np.exp(-1j * k * lorenz63_z_vals) for k in ks]).T
Psi = np.array([np.exp(-1j * k * lorenz63_train_timestamps) for k in ks]).T
w_vec = zeta_derivative(lorenz63_t_ref, lorenz63_train_timestamps, tau)
w_vec /= 2 * np.pi

ck = (lorenz63_f_vals * w_vec) @ (Psi)

# Normalize by the number of points
ck /= len(ck)
# Reconstruct F(z) using the Fourier coefficients
lorenz63_f_vals_hat = np.sum(
    [ck[m] * np.exp(1j * ks[m] * lorenz63_train_timestamps) for m in range(len(ks))],
    axis=0,
)
# Compare original and reconstructed functions
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(lorenz63_train_timestamps, lorenz63_f_vals, label="Original f(s)")

plt.plot(
    lorenz63_train_timestamps,
    np.real(lorenz63_f_vals_hat),
    label="Reconstructed f(s)",
    linestyle="--",
)
plt.legend()
plt.xlabel("s")
plt.ylabel("f(s)")
plt.title("Original vs Reconstructed Lorenz63 Function (when w(s) = 1)")
plt.grid(True)
plt.show()


# %%
# Plot the function in the s domain
plt.figure(figsize=(10, 6))
plt.plot(lorenz63_train_timestamps, np.real(lorenz63_train_data), label="Re[f(s)]")
plt.plot(
    lorenz63_train_timestamps,
    np.imag(lorenz63_train_data),
    label="Im[f(s)]",
    linestyle="--",
)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Lorenz63 Function in s domain")
plt.grid(True)
plt.legend()
plt.show()

# Plot the function in the z domain
plt.figure(figsize=(10, 6))
plt.plot(zvals, np.real(fz), label="Re[F(z)]")
plt.plot(zvals, np.imag(fz), label="Im[F(z)]", linestyle="--")
plt.xlabel("Time (z)")
plt.ylabel("Amplitude")
plt.title("Lorenz63 Function in z domain")
plt.grid(True)
plt.legend()
plt.show()

# %%
