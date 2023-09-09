import numpy as np
import pytest

from miv.core.datatype import Signal, Spikestamps
from miv.signal.spike import WaveformStatisticalFilter


def test_waveform_statistical_filter_normal():
    wave = np.zeros(10).reshape([10, 1])
    time = np.linspace(0, 1, 10)
    rate = 1.0 / 10
    testing_waveform = {0: Signal(wave, time, rate)}

    testing_spiketrains = Spikestamps()
    testing_spiketrains.append(np.array([0.5]))

    # Testing
    wsf = WaveformStatisticalFilter()
    result = wsf(testing_waveform, testing_spiketrains)

    np.testing.assert_allclose(result.data, [[0.5]])
    np.testing.assert_allclose(result.get_count(), [1])


def test_waveform_statistical_filter_mean_over():
    wave = np.zeros(10).reshape([10, 1]) + 5
    time = np.linspace(0, 1, 10)
    rate = 1.0 / 10
    testing_waveform = {0: Signal(wave, time, rate)}

    testing_spiketrains = Spikestamps()
    testing_spiketrains.append(np.array([0.5]))

    # Testing
    wsf = WaveformStatisticalFilter()
    result = wsf(testing_waveform, testing_spiketrains)

    np.testing.assert_allclose(result.get_count(), [0])


def test_waveform_statistical_filter_std_over():
    wave = (np.random.random(10) * 20).reshape([10, 1])
    time = np.linspace(0, 1, 10)
    rate = 1.0 / 10
    testing_waveform = {0: Signal(wave, time, rate)}

    testing_spiketrains = Spikestamps()
    testing_spiketrains.append(np.array([0.5]))

    # Testing
    wsf = WaveformStatisticalFilter()
    result = wsf(testing_waveform, testing_spiketrains)

    np.testing.assert_allclose(result.get_count(), [0])


def test_waveform_statistical_filter_mix():
    testing_waveform = {}

    wave1 = (
        np.array(
            [
                0.7624612,
                0.06174284,
                0.48327534,
                0.23766127,
                0.93223255,
                0.65475898,
                0.84306566,
                0.79945695,
                0.75793425,
                0.1380648,
            ]
        )
        * 5
    )
    wave2 = np.zeros(10)
    wave3 = np.zeros(10) + 10
    wave4 = np.zeros(10) - 3
    wave = np.vstack(
        [wave1, wave2, wave3, wave4]
    ).T  # TODO: remove T if axis change for signal
    time = np.linspace(0, 1, 10)
    rate = 1.0 / 10
    testing_waveform[0] = Signal(wave, time, rate)

    testing_spiketrains = Spikestamps()
    testing_spiketrains.append(np.array([0.5, 1.0, 1.5, 2.0]))

    # Testing
    wsf = WaveformStatisticalFilter(max_mean=5, max_std=1)
    result = wsf(testing_waveform, testing_spiketrains)

    np.testing.assert_allclose(result.data, [[1.0, 2.0]])
    np.testing.assert_allclose(result.get_count(), [2])
