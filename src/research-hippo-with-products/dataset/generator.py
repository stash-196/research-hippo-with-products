import os
import pickle
import numpy as np
import torch

# Plotly imports
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# For peak detection in power spectrum
from scipy.signal import find_peaks

###############################################################################
#                              WhiteNoise Generator
###############################################################################


def whitesignal(period, dt, freq, rms=0.5, batch_shape=()):
    """
    Produces output signal of length (period / dt), band-limited to frequency freq.
    Output shape = (*batch_shape, period/dt).

    Adapted from the nengo library.

    Parameters
    ----------
    period : float
        Total duration of the signal in real units (e.g., seconds).
    dt : float
        Timestep for discretization.
    freq : float
        Maximum frequency (cutoff) for band-limiting the white noise.
    rms : float, optional
        Desired root mean square of the signal, by default 0.5
    batch_shape : tuple, optional
        Extra shape to generate multiple signals at once, by default ()

    Returns
    -------
    signal : ndarray
        Band-limited white noise signal with shape (*batch_shape, period/dt).
    """

    # Check that freq >= 1/period to produce a non-zero signal
    if freq is not None and freq < 1.0 / period:
        raise ValueError(
            f"Make freq ({freq}) >= 1.0/period ({1.0/period}) to produce a non-zero signal."
        )

    # Nyquist frequency
    nyquist_cutoff = 0.5 / dt
    if freq > nyquist_cutoff:
        raise ValueError(
            f"{freq} exceeds Nyquist frequency {nyquist_cutoff:.3f} for dt={dt}."
        )

    n_coefficients = int(np.ceil(period / dt / 2.0))
    shape = batch_shape + (n_coefficients + 1,)

    # Random Fourier coefficients
    sigma = rms * np.sqrt(0.5)
    coefficients = 1j * np.random.normal(0.0, sigma, size=shape)
    coefficients += np.random.normal(0.0, sigma, size=shape)

    # Force DC component and highest freq to zero
    coefficients[..., 0] = 0.0
    coefficients[..., -1] = 0.0

    # Zero out frequencies above freq
    freqs = np.fft.rfftfreq(2 * n_coefficients, d=dt)
    set_to_zero = freqs > freq
    coefficients *= 1 - set_to_zero

    # Correct for lost power
    power_correction = np.sqrt(1.0 - np.sum(set_to_zero) / n_coefficients)
    if power_correction > 0.0:
        coefficients /= power_correction

    # Scale by sqrt(2 * n_coefficients) so RMS matches `rms`
    coefficients *= np.sqrt(2.0 * n_coefficients)

    # Inverse Fourier transform to get time-domain signal
    signal = np.fft.irfft(coefficients, axis=-1)

    # Remove DC offset by shifting start to zero
    signal = signal - signal[..., :1]

    return signal


class WhiteNoise:
    """
    A generator class for band-limited white noise, similar in structure to the L63 class.

    The entire noise is precomputed at initialization using `whitesignal()`.
    - `step()` returns one sample from this precomputed array
    - `integrate(n_steps)` calls `step()` repeatedly
    - The generated samples are stored in `self.hist`.
    """

    def __init__(self, period, dt, freq, rms=0.5, batch_shape=()):
        """
        Parameters
        ----------
        period : float
            Total duration (in real units) for the generated noise.
        dt : float
            Time step size.
        freq : float
            Maximum frequency cutoff for band-limiting.
        rms : float, optional
            Desired RMS for the noise, by default 0.5.
        batch_shape : tuple, optional
            Extra shape to generate multiple signals (not shown here), by default ().
        """
        self.period = period
        self.dt = dt
        self.freq = freq
        self.rms = rms
        self.batch_shape = batch_shape

        # Generate full noise signal for the entire duration
        self._signal = whitesignal(
            period=self.period,
            dt=self.dt,
            freq=self.freq,
            rms=self.rms,
            batch_shape=self.batch_shape,
        )

        # Index pointer into the precomputed signal
        self._index = 0

        # Holds samples produced by step()
        self.hist = []

    def step(self):
        """Returns the next sample from the precomputed noise and appends it to `hist`."""
        if self._index >= self._signal.shape[-1]:
            self._index = 0  # Wrap around if we prefer cyclical behavior

        sample = self._signal[..., self._index]
        self.hist.append(sample)
        self._index += 1
        return sample

    def integrate(self, n_steps):
        """Calls `step()` n_steps times."""
        for _ in range(n_steps):
            self.step()


###############################################################################
#                         Lorenz 63 (L63) System
###############################################################################


class L63:
    """
    Implements the Lorenz 63 system with methods to step through
    and store the trajectory history (similar structure to WhiteNoise).
    """

    def __init__(self, sigma, rho, beta, init, dt):
        """
        Parameters
        ----------
        sigma, rho, beta : float
            Lorenz 63 parameters
        init : list or tuple
            Initial state [x, y, z]
        dt : float
            Time step size
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.x, self.y, self.z = init
        self.dt = dt
        self.hist = [init]

    def step(self):
        """
        One Euler step of the Lorenz 63 system, appending the new state to `hist`.
        """
        self.x += self.sigma * (self.y - self.x) * self.dt
        self.y += (self.x * (self.rho - self.z)) * self.dt
        self.z += (self.x * self.y - self.beta * self.z) * self.dt
        self.hist.append([self.x, self.y, self.z])

    def integrate(self, n_steps):
        """Repeatedly calls step()."""
        for _ in range(n_steps):
            self.step()


###############################################################################
#                              Plotting Functions
###############################################################################


def plot_attractor_plotly(hists, save_dir=None, explain=None, format="pdf"):
    """
    Plots one or more Lorenz 63 trajectories in 3D using Plotly.
    If 'save_dir' is specified, saves the figure.
    """
    if np.array(hists).ndim == 2:
        hists = [hists]
    hists = [np.array(h) for h in hists]

    fig = go.Figure()
    for h in hists:
        fig.add_trace(
            go.Scatter3d(
                x=h[:, 0],
                y=h[:, 1],
                z=h[:, 2],
                mode="lines",
                line=dict(color="blue"),
            )
        )

    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        title=f"Attractor Plot for {explain} set",
    )
    fig.show()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"attractor_{explain}.{format}")
        pio.write_image(fig, save_path)


def plot_attractor_subplots(hists, explain, save_dir=None, format="pdf"):
    """
    Plots timeseries of x, y, and z in vertical subplots using Plotly.
    If 'save_dir' is given, saves the figure.
    """
    if np.array(hists).ndim == 2:
        hists = [hists]
    hists = [np.array(h) for h in hists]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("X Timeseries", "Y Timeseries", "Z Timeseries"),
    )

    for h in hists:
        fig.add_trace(
            go.Scatter(y=h[:, 0], mode="lines", line=dict(color="blue")), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=h[:, 1], mode="lines", line=dict(color="red")), row=2, col=1
        )
        fig.add_trace(
            go.Scatter(y=h[:, 2], mode="lines", line=dict(color="green")), row=3, col=1
        )

    fig.update_layout(
        title_text=f"Timeseries Subplots for X, Y, and Z for {explain} set"
    )
    fig.show()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"timeseries_{explain}.{format}")
        pio.write_image(fig, save_path)


def plot_components_vs_time_plotly(
    time_series, time_step, explain, save_dir=None, format="pdf"
):
    """
    Plots x, y, and z components of a time series (like L63) vs time with Plotly.
    """
    t = np.arange(0, len(time_series) * time_step, time_step)
    x, y, z = time_series[:, 0], time_series[:, 1], time_series[:, 2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, mode="lines", name="x", line=dict(color="blue")))
    fig.add_trace(
        go.Scatter(x=t, y=y, mode="lines", name="y", line=dict(color="green"))
    )
    fig.add_trace(go.Scatter(x=t, y=z, mode="lines", name="z", line=dict(color="red")))

    fig.update_layout(
        title=f"Components of Lorenz63 System vs. Time for {explain} set",
        xaxis_title="Time",
        yaxis_title="Values",
        showlegend=True,
        template="plotly_white",
    )
    fig.show()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"components_vs_time_{explain}.{format}")
        pio.write_image(fig, save_path)


def calculate_power_spectrum(time_series, sampling_rate):
    """
    Computes power spectrum of a 1D time series using FFT.
    Returns time_periods, frequencies, and power_spectrum for positive frequencies.
    """
    fft_result = np.fft.fft(time_series)
    freqs = np.fft.fftfreq(len(time_series), 1 / sampling_rate)

    # Time periods (1/frequency) only for nonzero frequencies
    time_periods = np.zeros_like(freqs)
    with np.errstate(divide="ignore"):
        time_periods[1:] = 1.0 / freqs[1:]

    # Keep only positive frequencies
    nonzero_indices = np.where(freqs > 0)
    power_spectrum = np.abs(fft_result[nonzero_indices]) ** 2
    frequencies = freqs[nonzero_indices]

    return time_periods, frequencies, power_spectrum


def plot_power_spectrum_plotly(
    time_series, sampling_rate, explain, save_dir=None, format="pdf"
):
    """
    Plots the power spectrum of x, y, z components on a log scale in Plotly.
    """
    fig = go.Figure()

    for i, component in enumerate(["x", "y", "z"]):
        series = np.array(time_series)[:, i]
        time_periods, frequencies, spectrum = calculate_power_spectrum(
            series, sampling_rate
        )

        # Skip the zero-frequency bin
        time_periods, frequencies, spectrum = (
            time_periods[1:],
            frequencies[1:],
            spectrum[1:],
        )

        fig.add_trace(
            go.Scatter(
                x=time_periods,
                y=spectrum,
                mode="lines",
                name=f"{component} power spectrum",
            )
        )

    fig.update_layout(
        title=f"Power Spectrum of Lorenz63 System for {explain} set",
        xaxis=dict(type="log", title="Time Periods"),
        yaxis=dict(title="Power"),
        template="plotly_white",
    )
    fig.show()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"power_spectrum_{explain}.{format}")
        pio.write_image(fig, save_path)


def plot_power_spectrum_subplots_loglog(
    time_series, sampling_rate, explain, component_labels, save_dir=None, format="pdf"
):
    """
    Plots the power spectrum of each component on a log-log scale, including peak detection.
    Suitable for both original data (XYZ) and PCA-transformed data.

    :param time_series: (n_samples, n_components) array
    :param sampling_rate: sampling rate (e.g., 1/dt)
    :param explain: str label for the dataset
    :param component_labels: labels for each component, e.g. ["X", "Y", "Z"]
    """
    fig = make_subplots(
        rows=len(component_labels),
        cols=1,
        subplot_titles=[f"{label} Power Spectrum" for label in component_labels],
    )

    for i, label in enumerate(component_labels):
        series = time_series[:, i]
        fft_result = np.fft.fft(series)
        freqs = np.fft.fftfreq(len(series), 1 / sampling_rate)
        power_spectrum = np.abs(fft_result) ** 2

        # Filter to positive frequencies
        positive_freqs = freqs > 0
        frequencies = freqs[positive_freqs]
        power_spectrum = power_spectrum[positive_freqs]

        # Detect peaks
        peaks, properties = find_peaks(
            power_spectrum, height=0.5, distance=100, prominence=0.05
        )
        peak_heights = properties["peak_heights"]

        # Take top 3 peaks
        if len(peak_heights) > 0:
            largest_peaks_indices = np.argsort(peak_heights)[-3:]
            peaks = peaks[largest_peaks_indices]

        # Plot the spectrum
        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=power_spectrum,
                mode="lines",
                name=f"{label} Power Spectrum",
            ),
            row=i + 1,
            col=1,
        )
        # Mark the peaks
        if len(peaks) > 0:
            fig.add_trace(
                go.Scatter(
                    x=frequencies[peaks],
                    y=power_spectrum[peaks],
                    mode="markers",
                    marker=dict(color="red", size=8),
                    name=f"{label} Peaks",
                ),
                row=i + 1,
                col=1,
            )

    fig.update_layout(
        height=600,
        width=800,
        title_text=f"Power Spectrum for {explain} (Log-Log Scale)",
    )
    fig.update_xaxes(type="log", title="Frequency")
    fig.update_yaxes(type="log", title="Power Spectrum")

    fig.show()

    if save_dir is not None and format:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"power_spectrum_{explain}.{format}")
        pio.write_image(fig, save_path)

    return peaks, frequencies, power_spectrum


def save_pickle(data, path):
    """Saves `data` as a pickle file at the given `path`."""
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved data to {path}")


def plot_delay_embedding(observation, delay, dimensions):
    """
    Plots the delay embedding of a 1D observation using Plotly.
    Supports 2D and 3D embeddings.
    """
    n = len(observation)
    embedding_length = n - (dimensions - 1) * delay
    if embedding_length <= 0:
        raise ValueError(
            "Delay and dimensions are too large for the length of the observation array."
        )

    embedded = np.empty((embedding_length, dimensions))
    for i in range(dimensions):
        embedded[:, i] = observation[i * delay : i * delay + embedding_length]

    if dimensions == 2:
        fig = go.Figure(
            data=go.Scatter(x=embedded[:, 0], y=embedded[:, 1], mode="lines")
        )
        fig.update_layout(
            title="2D Delay Embedding",
            xaxis_title="X(t)",
            yaxis_title="X(t + delay)",
        )
        fig.show()

    elif dimensions == 3:
        fig = go.Figure(
            data=go.Scatter3d(
                x=embedded[:, 0],
                y=embedded[:, 1],
                z=embedded[:, 2],
                mode="lines",
            )
        )
        fig.update_layout(
            title="3D Delay Embedding",
            scene=dict(
                xaxis_title="X(t)",
                yaxis_title="X(t + delay)",
                zaxis_title="X(t + 2 * delay)",
            ),
        )
        fig.show()

    else:
        raise NotImplementedError(
            "Plotting for dimensions higher than 3 is not implemented."
        )


###############################################################################
#                                 Main Demo
###############################################################################

if __name__ == "__main__":
    ###########################################################################
    # Example 1: WhiteNoise usage
    ###########################################################################
    wn = WhiteNoise(period=10.0, dt=0.01, freq=5.0, rms=0.5)
    wn.integrate(100)  # take 100 steps
    print("First 10 WhiteNoise samples:", wn.hist[:10])

    ###########################################################################
    # Example 2: Lorenz 63 usage
    ###########################################################################
    # Create directories (example)
    file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir_plots = os.path.join(file_dir, "temp_save", "lorenz63", "plots")
    save_dir_data = os.path.join(file_dir, "temp_save", "lorenz63", "data")
    os.makedirs(save_dir_plots, exist_ok=True)
    os.makedirs(save_dir_data, exist_ok=True)

    # Define Lorenz 63 parameters and generate data
    sigma = 10
    rho = 28
    beta = 8 / 3
    N = 1000  # e.g., 1000 steps
    dt = 1e-2

    l1 = L63(sigma, rho, beta, init=[1, 10, 20], dt=dt)
    l2 = L63(sigma, rho, beta, init=[10, 1, 2], dt=dt)

    l1.integrate(N)
    l2.integrate(int(N * 0.1))

    # Plot a 3D attractor
    plot_attractor_plotly(
        [l1.hist], save_dir=save_dir_plots, explain="s10_r28_b8d3_train"
    )

    # Plot subplots of X, Y, Z over time
    plot_attractor_subplots(
        [l1.hist], explain="s10_r28_b8d3_train", save_dir=save_dir_plots
    )

    # Plot components vs. time
    plot_components_vs_time_plotly(
        np.array(l1.hist),
        time_step=dt,
        explain="s10_r28_b8d3_train",
        save_dir=save_dir_plots,
    )

    # Save data
    save_pickle(l1.hist, os.path.join(save_dir_data, "dataset_train.pkl"))

    # Calculate sampling rate (1/dt)
    sampling_rate = 1 / dt

    # Plot power spectrum
    plot_power_spectrum_plotly(
        np.array(l1.hist), sampling_rate, "s10_r28_b8d3_train", save_dir=save_dir_plots
    )

    # Plot power spectrum (log-log) with peak detection
    component_labels_xyz = ["X", "Y", "Z"]
    peaks_train, freqs_train, ps_train = plot_power_spectrum_subplots_loglog(
        np.array(l1.hist),
        sampling_rate,
        "s10_r28_b8d3_train_xyz",
        component_labels_xyz,
        save_dir=save_dir_plots,
    )
    peaks_test, freqs_test, ps_test = plot_power_spectrum_subplots_loglog(
        np.array(l2.hist),
        sampling_rate,
        "s10_r28_b8d3_test_xyz",
        component_labels_xyz,
        save_dir=save_dir_plots,
    )

    print("Finished Lorenz and WhiteNoise demo!")
