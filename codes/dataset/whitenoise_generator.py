import numpy as np


def whitesignal(period, dt, freq, rms=0.5, batch_shape=()):
    """
    Produces output signal of length (period / dt), band-limited to frequency freq.
    Output shape = (*batch_shape, period/dt).

    Adapted from the nengo library.
    """
    # Check freq >= 1.0/period to get a non-zero signal
    if freq is not None and freq < 1.0 / period:
        raise ValueError(
            f"Make freq ({freq}) >= 1.0/period ({1.0/period}) to produce a non-zero signal."
        )

    # Check Nyquist
    nyquist_cutoff = 0.5 / dt
    if freq > nyquist_cutoff:
        raise ValueError(
            f"{freq} exceeds Nyquist frequency {nyquist_cutoff:.3f} for dt={dt}."
        )

    n_coefficients = int(np.ceil(period / dt / 2.0))
    shape = batch_shape + (n_coefficients + 1,)

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

    # Scale so RMS matches `rms`
    coefficients *= np.sqrt(2.0 * n_coefficients)

    # Inverse Fourier transform -> time domain
    signal = np.fft.irfft(coefficients, axis=-1)
    # Remove DC offset by shifting start to zero
    signal = signal - signal[..., :1]

    return signal


class WhiteNoise:
    """
    A generator for band-limited white noise, similar in structure to the L63 class.
    - `step()` returns one sample
    - `integrate(n_steps)` calls step() multiple times
    - `hist` stores the generated samples
    """

    def __init__(self, period, dt, freq, rms=0.5, batch_shape=()):
        self.period = period
        self.dt = dt
        self.freq = freq
        self.rms = rms
        self.batch_shape = batch_shape

        self._signal = whitesignal(period, dt, freq, rms, batch_shape)
        self._index = 0
        self.hist = []

    def step(self):
        """
        Returns the next sample from the precomputed white noise,
        wrapping around if we've reached the end.
        """
        if self._index >= self._signal.shape[-1]:
            self._index = 0  # Wrap if desired

        sample = self._signal[..., self._index]
        self.hist.append(sample)
        self._index += 1
        return sample

    def integrate(self, n_steps):
        """Calls step() n_steps times."""
        for _ in range(n_steps):
            self.step()
