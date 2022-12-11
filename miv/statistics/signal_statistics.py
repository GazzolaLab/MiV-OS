__all__ = ["signal_to_noise", "spike_amplitude_to_background_noise"]

import numpy as np

from miv.typing import SignalType, SpikestampsType
from miv.visualization import extract_waveforms


def signal_to_noise(signal: SignalType, axis: int = 0, ddof: int = 0):
    """
    Compute signal-to-noise ratio of raw signal.

    Parameters
    ----------
    signal : SignalType
    axis : int
        Axis of interest. By default, signal axis is 0 (default=1)
    ddof : int
    """
    signal_np = np.asanyarray(signal)
    m = signal_np.mean(axis)
    sd = signal_np.std(axis=axis, ddof=ddof)
    return np.abs(np.where(sd == 0, 0, m / sd))


def spike_amplitude_to_background_noise(
    signal: SignalType, spikestamps: SpikestampsType, sampling_rate: float
):
    """
    Given the signal (filtered) and spikestamps, compute the signal-to-noise ratio where signal is the
    average power of the spike and the noise is the power of the background noise.

    :ref:`<https://en.wikipedia.org/wiki/Signal-to-noise_ratio>`

    Parameters
    ----------
    signal: SignalType,
        The signal as a 2-dimensional numpy array (length, num_channel)
    spikestamps: SpikestampsType,
        The sample index of all spikes as a 1-dim numpy array
    sampling_rate : float
        The sampling frequency in Hz
    """
    assert signal.shape[1] == len(
        spikestamps
    ), f"The number of channel for given signal {signal.shape[1]} is not equal to the number of channels in spikestamps {len(spikestamps)}."

    snr = []
    num_channels = signal.shape[1]
    for channel in range(num_channels):
        # Compute mean spike amplitude
        cutouts = extract_waveforms(signal, spikestamps, channel, sampling_rate)
        mean_spike_amplitude = np.mean([np.max(np.abs(cutout)) for cutout in cutouts])

        # Compute background noise
        signal_np = np.asanyarray(signal[:, channel])
        background_noise_amplitude = signal_np.std()

        snr.append((mean_spike_amplitude / background_noise_amplitude) ** 2)

    return snr
