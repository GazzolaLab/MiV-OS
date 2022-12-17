import numpy as np
import pytest

from miv.statistics.signal_statistics import (
    signal_to_noise,
    spike_amplitude_to_background_noise,
)
from miv.typing import SignalType


def test_signal_to_noise():
    signal = np.zeros(10)
    signal[3] = 1.0
    assert signal_to_noise(signal) == 1.0 / 3.0

    signal = np.zeros(10)
    signal[6] = 1.0
    assert signal_to_noise(signal) == 1.0 / 3.0

    signal = np.zeros(10)
    signal[3] = 1.0
    signal[6] = 1.0
    assert np.isclose(signal_to_noise(signal), 1.0 / 2.0)


def test_spike_amplitude_to_background_noise_arbitrary_signal_and_timestamps():
    sampling_rate = 1000

    # Test a single channel with no spikes
    signal = np.array([[1, 2, 3, 4, 5]]).T
    spikestamps = np.array([[]])
    expected = np.array([np.nan])
    actual = spike_amplitude_to_background_noise(signal, spikestamps, sampling_rate)
    assert np.allclose(expected, actual, equal_nan=True)

    # Test a single channel with one spike
    signal = np.array([[1, 2, 3, 4, 5]]).T
    spikestamps = np.array([[2]]) / sampling_rate
    expected = np.array([8])
    actual = spike_amplitude_to_background_noise(signal, spikestamps, sampling_rate)
    assert np.allclose(expected, actual)

    # Test multiple channels with no spikes
    signal = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).T
    spikestamps = np.array([[], []])
    expected = np.array([np.nan, np.nan])
    actual = spike_amplitude_to_background_noise(signal, spikestamps, sampling_rate)
    assert np.allclose(expected, actual, equal_nan=True)

    # Test multiple channels with different spikes
    signal = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).T
    spikestamps = np.array([[2], [4]]) / sampling_rate
    expected = np.array([8.0, 50.0])
    actual = spike_amplitude_to_background_noise(signal, spikestamps, sampling_rate)
    assert np.allclose(expected, actual)


def test_spike_amplitude_to_background_noise_assertion_error():
    # Create a test signal with 2 channels and 10 samples per channel
    signal = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]
    ).T

    # Set the sampling rate to 1000 Hz
    sampling_rate = 1000

    # Create spikestamps for each channel with 2 spikes per channel
    spikestamps = np.array([[1, 4], [7, 9]]) / sampling_rate

    # Create an additional spikestamp channel
    spikestamps_2 = np.concatenate((spikestamps, [[1, 4]]), axis=0)

    # Check that the assertion error is raised when the number of channels in the signal
    # does not match the number of channels in spikestamps
    with pytest.raises(AssertionError):
        spike_amplitude_to_background_noise(signal, spikestamps_2, sampling_rate)
