# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from research_hippo_with_products.utils import find_project_root

def plot_basis_functions(t, basis_matrix, title):
    """
    Plot the basis functions.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(t, basis_matrix.real)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Basis Value")
    plt.show()

def plot_reconstructed_signal(time, signal, reconstructed_signal, title):
    """
    Plot the original and reconstructed signals.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(time, signal.real, label='Original Signal (Real Part)')
    plt.plot(time, reconstructed_signal.real, '--', label='Reconstructed Signal (Real Part)')
    plt.title(title)
    plt.xlabel("Time")
    plt.legend()
    plt.show()

def plot_signal(time, signal, title, xlabel="Time"):
    """
    Plot the original signal.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(time, signal.real)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.show()

# --- 1. Basis Functions ---
def compute_basis(x, ks, basis_type="fourier"):
    """
    Compute the basis matrix for a given basis type.
    
    Parameters:
      - x: Array of domain values (e.g., time)
      - ks: Array of basis indices (e.g., Fourier indices)
      - basis_type: Type of basis ("fourier", "polynomial", "legendre", etc.)
    
    Returns:
      - Basis matrix of shape (len(x), len(ks))
    """
    if basis_type == "fourier":
        # Create a Fourier basis. Note that the 1/sqrt(N) normalization
        # makes the basis orthonormal with respect to the discrete inner product.
        return np.exp(-1j * 2 * np.pi * np.outer(ks, x)).T
    else:
        raise ValueError(f"Unsupported basis type: {basis_type}")

# --- 2. Projection (Compute Coefficients) ---
def compute_coefficients(signal, time, ks, weights, basis_type="fourier"):
    """
    Compute the coefficients for a given signal using the chosen basis.
    
    Parameters:
      - signal: The input signal (as a numpy array)
      - time: Array of time (or spatial) samples
      - ks: Array of basis indices
      - weights: Weights for the signal (e.g., ones if uniform)
      - basis_type: Type of basis to use
    
    Returns:
      - ck: Coefficients obtained by projecting the signal onto the basis
      - Psi: The basis matrix evaluated at time
    """
    # Use the time array as the domain for basis evaluation
    Psi = compute_basis(time, ks, basis_type)
    
    # If the time grid is uniform, delta_s is the spacing between samples.
    # (If non-uniform, you might want to pass in the appropriate integration weights.)
    delta_s = time[1] - time[0] if len(time) > 1 else 1
    T = time[-1] - time[0]
    print("T: ", T)
    print("delta_s: ", delta_s)
    print("len(time): ", len(time))    
    # Compute the coefficients using a discrete inner product.
    # (Note: Depending on your convention you might need to use the conjugate of Psi here.)
    print("T/delta_s vs len(time): ", T/delta_s, " : ", len(time))
    print("1/N: ", 1/len(time))
    Psi_norm = Psi / len(time)
    ck = signal @ Psi_norm
    print("ck magnitude of DC: ", np.abs(ck)[np.where(ks == 0)[0]])
    return ck, Psi

# --- 3. Reconstruction and Plotting ---
def vanilla_process_signal(time, signal, ks, basis_type, title):
    """
    Process a signal by projecting onto a basis and then reconstructing.
    
    Parameters:
      - time: Array of time samples
      - signal: The original signal
      - t_ref: A reference time (e.g., used for alignment, if needed)
      - K: Number of basis functions to use
      - tau: (Optional) additional parameter (not used in this basic example)
      - basis_type: The type of basis to use ("fourier", etc.)
      - title: Title for the plot
    """
    
    # For simplicity, use uniform weights.
    weights = np.ones_like(signal)
    
    # Project the signal onto the basis.
    ck, Psi = compute_coefficients(signal, time, ks, weights, basis_type)
    
    # Reconstruct the signal using the conjugate of the basis.
    # Since our forward projection did not conjugate the basis, we do it here.
    T = time[-1] - time[0]

    reconstructed_signal = Psi.conj() @ ck

    print("mean of original signal: ", np.mean(signal.real))
    print("mean of reconstructed signal: ", np.mean(reconstructed_signal.real))
    print("scale factor: ", np.mean(reconstructed_signal.real) /np.mean(signal.real))
    # std
    print("std of original signal: ", np.std(signal.real))
    print("std of reconstructed signal: ", np.std(reconstructed_signal.real))
    print("scale factor: ", np.std(reconstructed_signal.real) /np.std(signal.real))
    # max and min
    print("max and min of original signal: ", np.max(signal.real), np.min(signal.real))
    print("max and min of reconstructed signal: ", np.max(reconstructed_signal.real), np.min(reconstructed_signal.real))
    print("scale factor: ", np.max(reconstructed_signal.real) / np.max(signal.real), np.min(reconstructed_signal.real) / np.min(signal.real))
    
    if basis_type == "fourier":
        Psi_plot = Psi[:, np.where(ks >= 0)[0]].copy()
        # reverse order of Psi_plot
        Psi_plot = Psi_plot[:, ::-1]
    plot_basis_functions(time, Psi_plot, f"{title} Basis Functions")
    plot_reconstructed_signal(time, signal, reconstructed_signal, title)
    # plot the coefficients
    plot_signal(ks, np.abs(ck), f"{title} Coefficients", xlabel="Fourier Index")


    
# %%
# create toy data of a sine wave with 3 frequencies, sampled at 200 Hz for 4 seconds
sinusoid_t = np.linspace(0, 4, 800)
len(sinusoid_t)
f1 = 1
f2 = 10
f3 = 20
sinusoid_data = np.sin(2 * np.pi * f1 * sinusoid_t) + np.sin(2 * np.pi * f2 * sinusoid_t) + np.sin(2 * np.pi * f3 * sinusoid_t) + 0.5

N = len(sinusoid_data)
T = sinusoid_t[1] - sinusoid_t[0]
ks = np.arange(-N//2+1, N//2)/N/T

vanilla_process_signal(
    sinusoid_t,
    sinusoid_data,
    ks,
    "fourier",
    "Sinusoid Signal Reconstruction"
)


#%%
# Load data
root_dir = find_project_root(__file__)
data_dir = os.path.join(root_dir, "data")

whitenoise_data_path = os.path.join(data_dir, "whitenoise_samples.pkl")

with open(whitenoise_data_path, "rb") as f:
    whitenoise_data = pickle.load(f)

# Process the whitenoise signal.
# Here we reconstruct only the first half of the dataset.
whitenoise_half_point = len(whitenoise_data["time"]) // 2
whitenoise_t_ref = whitenoise_data["time"][whitenoise_half_point]
whitenoise_train_time = whitenoise_data["time"][:whitenoise_half_point]
whitenoise_train_data = np.asarray(
    whitenoise_data["data"][:whitenoise_half_point], dtype=np.complex128
)

# Standardize the signal
if False: 
    whitenoise_train_data = (whitenoise_train_data - np.mean(whitenoise_train_data) ) / np.std(whitenoise_train_data)

whitenoise_train_data = whitenoise_train_data - np.mean(whitenoise_train_data) + 5

N = len(whitenoise_train_data)
T = whitenoise_train_time[1] - whitenoise_train_time[0]
ks = np.arange(-N//2+1, N//2)/N/T
# Run the processing function to project and reconstruct the signal.
vanilla_process_signal(
    whitenoise_train_time,
    whitenoise_train_data,
    ks,
    "fourier",
    "Whitenoise Signal Reconstruction"
)
#%%
# use package to get fourier transform of data and plot it so we can compare
# the coefficients
from scipy.fft import fft, fftfreq
# Number of samples
N = len(whitenoise_train_data)
# Sample spacing
T = whitenoise_train_time[1] - whitenoise_train_time[0]
# Get the fourier transform
yf = fft(whitenoise_train_data)
xf = fftfreq(N, T)[:N//2]
print("xf: ", xf)
# Plot the fourier transform
plt.figure(figsize=(10, 5))
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.title("Whitenoise Signal Fourier Transform")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.show()

# use reverse Fourier transform to get the signal back
from scipy.fft import ifft
reconstructed_signal = ifft(yf)
# Plot the original and reconstructed signals
plt.figure(figsize=(10, 5))
plt.plot(whitenoise_train_time, whitenoise_train_data.real, label='Original Signal (Real Part)')
plt.plot(whitenoise_train_time, reconstructed_signal.real, '--', label='Reconstructed Signal (Real Part)')
plt.title("Whitenoise Signal Reconstruction")
plt.xlabel("Time")
plt.legend()
plt.show()

# %%
