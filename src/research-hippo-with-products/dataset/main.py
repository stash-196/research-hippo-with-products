import os
import numpy as np

# Local imports
from whitenoise_generator import WhiteNoise
from lorenz63_generator import L63
import plot_utils as pu  # rename as you like


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # 1. WhiteNoise Demo
    # ----------------------------------------------------------------
    wn = WhiteNoise(period=10.0, dt=0.01, freq=5.0, rms=0.5)
    wn.integrate(2000)
    print("First 10 WhiteNoise samples:", wn.hist[:10])

    # Plot 1D white noise
    pu.plot_1d_series(
        time_series=np.array(wn.hist),
        time_step=0.01,
        explain="white_noise_demo",
        save_dir="codes/dataset/temp_save/whitenoise/plots",
    )

    # Create a timestamps array for the white noise
    wn_timestamps = np.arange(0, len(wn.hist) * 0.01, 0.01)

    # Save the WhiteNoise data AND timestamps
    whitenoise_save_path = os.path.join(
        "codes/dataset/temp_save/whitenoise/data",
        "whitenoise_samples.pkl",
    )
    os.makedirs(os.path.dirname(whitenoise_save_path), exist_ok=True)

    pu.save_pickle(
        {
            "time": wn_timestamps,
            "data": wn.hist,
        },
        whitenoise_save_path,
    )

    # ----------------------------------------------------------------
    # 2. Lorenz63 Demo
    # ----------------------------------------------------------------
    sigma = 10
    rho = 28
    beta = 8 / 3
    N = 2000
    dt = 1e-2

    # Create an instance
    l1 = L63(sigma, rho, beta, init=[1, 10, 20], dt=dt)
    l2 = L63(sigma, rho, beta, init=[10, 1, 2], dt=dt)

    l1.integrate(N)
    l2.integrate(int(N * 0.1))

    # Make some directories
    file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir_plots = os.path.join(file_dir, "temp_save", "lorenz63", "plots")
    save_dir_data = os.path.join(file_dir, "temp_save", "lorenz63", "data")
    os.makedirs(save_dir_plots, exist_ok=True)
    os.makedirs(save_dir_data, exist_ok=True)

    # Plot a 3D attractor
    pu.plot_attractor_plotly(
        [l1.hist], save_dir=save_dir_plots, explain="s10_r28_b8d3_train"
    )

    # Plot subplots of X, Y, Z
    pu.plot_attractor_subplots(
        [l1.hist], explain="s10_r28_b8d3_train", save_dir=save_dir_plots
    )

    # Plot components vs time
    pu.plot_components_vs_time_plotly(
        np.array(l1.hist), dt, explain="s10_r28_b8d3_train", save_dir=save_dir_plots
    )

    # Create timestamps for the Lorenz data
    l1_timestamps = np.arange(0, len(l1.hist) * dt, dt)

    # Save data (Lorenz)
    train_data_path = os.path.join(save_dir_data, "dataset_train.pkl")
    pu.save_pickle(
        {
            "time": l1_timestamps,
            "data": l1.hist,
        },
        train_data_path,
    )

    # Frequency analysis
    sampling_rate = 1 / dt

    # Power spectrum
    pu.plot_power_spectrum_plotly(
        np.array(l1.hist),
        sampling_rate,
        explain="s10_r28_b8d3_train",
        save_dir=save_dir_plots,
    )

    # Power spectrum subplots with log-log and peak detection
    component_labels_xyz = ["X", "Y", "Z"]
    pu.plot_power_spectrum_subplots_loglog(
        np.array(l1.hist),
        sampling_rate,
        explain="s10_r28_b8d3_train_xyz",
        component_labels=component_labels_xyz,
        save_dir=save_dir_plots,
    )

    # Example: delay embedding of X
    x = np.array(l1.hist)[:, 0]
    pu.plot_delay_embedding(x, delay=10, dimensions=3)

    print("Done with WhiteNoise + Lorenz63 demonstration!")
