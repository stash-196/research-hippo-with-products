import os
import pickle
import numpy as np

import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.signal import find_peaks


###############################################################################
#                               Saving / Loading
###############################################################################


def save_pickle(data, path):
    """
    Saves `data` as a pickle file at the given path.
    """
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved data to {path}")


###############################################################################
#                              Plotting
###############################################################################


def plot_attractor_plotly(hists, save_dir=None, explain=None, format="pdf"):
    """
    Plots one or more 3D trajectories (like Lorenz 63).
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


def plot_attractor_subplots(hists, explain="", save_dir=None, format="pdf"):
    """
    Plots time series of x, y, z on stacked subplots.
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

    colors = ["blue", "red", "green"]
    for h in hists:
        for row_i in range(3):
            fig.add_trace(
                go.Scatter(y=h[:, row_i], mode="lines", line=dict(color=colors[row_i])),
                row=row_i + 1,
                col=1,
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
    time_series, time_step, explain="", save_dir=None, format="pdf"
):
    """
    Plots x, y, z vs time with Plotly.
    """
    t = np.arange(0, len(time_series) * time_step, time_step)
    x, y, z = time_series[:, 0], time_series[:, 1], time_series[:, 2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, mode="lines", name="x", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="y", line=dict(color="red")))
    fig.add_trace(
        go.Scatter(x=t, y=z, mode="lines", name="z", line=dict(color="green"))
    )

    fig.update_layout(
        title=f"Components vs. Time for {explain} set",
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


###############################################################################
#                             Power Spectrum
###############################################################################


def calculate_power_spectrum(time_series, sampling_rate):
    """
    Compute power spectrum of a 1D series using FFT.
    Returns time_periods, frequencies, power_spectrum for positive frequencies.
    """
    fft_result = np.fft.fft(time_series)
    freqs = np.fft.fftfreq(len(time_series), 1 / sampling_rate)

    time_periods = np.zeros_like(freqs)
    with np.errstate(divide="ignore"):
        time_periods[1:] = 1.0 / freqs[1:]

    # Keep only positive frequencies
    idx = freqs > 0
    power_spectrum = np.abs(fft_result[idx]) ** 2
    frequencies = freqs[idx]

    return time_periods, frequencies, power_spectrum


def plot_power_spectrum_plotly(
    time_series, sampling_rate, explain="", save_dir=None, format="pdf"
):
    """
    Plots the power spectrum (x, y, z) on a log scale in Plotly.
    """
    fig = go.Figure()
    components = ["x", "y", "z"]

    for i, comp in enumerate(components):
        data_1d = time_series[:, i]
        time_periods, freqs, spectrum = calculate_power_spectrum(data_1d, sampling_rate)
        # skip index 0
        time_periods, freqs, spectrum = time_periods[1:], freqs[1:], spectrum[1:]

        fig.add_trace(
            go.Scatter(
                x=time_periods,
                y=spectrum,
                mode="lines",
                name=f"{comp} power spectrum",
            )
        )

    fig.update_layout(
        title=f"Power Spectrum for {explain}",
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
    Plots power spectrum on a log-log scale for each component, with peak detection.
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
        idx = freqs > 0
        frequencies = freqs[idx]
        power_spectrum = power_spectrum[idx]

        # Detect peaks
        peaks, properties = find_peaks(
            power_spectrum, height=0.5, distance=100, prominence=0.05
        )
        peak_heights = properties["peak_heights"]

        # Top 3 peaks
        if len(peak_heights) > 0:
            largest_peaks_indices = np.argsort(peak_heights)[-3:]
            peaks = peaks[largest_peaks_indices]

        # Plot
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
        # Plot peaks
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


###############################################################################
#                        Delay Embedding for 1D Series
###############################################################################


def plot_delay_embedding(observation, delay, dimensions):
    """
    Plots a 2D or 3D delay embedding for a 1D observation.
    """
    import plotly.graph_objects as go  # to limit top-level imports

    n = len(observation)
    embedding_length = n - (dimensions - 1) * delay
    if embedding_length <= 0:
        raise ValueError("Delay & dimensions too large for length of observation.")

    embedded = np.empty((embedding_length, dimensions))
    for i in range(dimensions):
        embedded[:, i] = observation[i * delay : i * delay + embedding_length]

    if dimensions == 2:
        fig = go.Figure(
            data=go.Scatter(x=embedded[:, 0], y=embedded[:, 1], mode="lines")
        )
        fig.update_layout(
            title="2D Delay Embedding", xaxis_title="X(t)", yaxis_title="X(t + delay)"
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
                zaxis_title="X(t + 2*delay)",
            ),
        )
        fig.show()

    else:
        raise NotImplementedError("Plotting for dims > 3 not implemented.")


def plot_1d_series(time_series, time_step, explain="", save_dir=None, format="pdf"):
    """
    Plots a single 1D time_series using Plotly.
    """
    import plotly.graph_objects as go
    import numpy as np

    t = np.arange(0, len(time_series) * time_step, time_step)
    fig = go.Figure(data=go.Scatter(x=t, y=time_series, mode="lines"))
    fig.update_layout(
        title=f"1D Time Series Plot for {explain}",
        xaxis_title="Time",
        yaxis_title="Signal",
        template="plotly_white",
    )
    fig.show()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"1d_timeseries_{explain}.{format}")
        pio.write_image(fig, save_path)
