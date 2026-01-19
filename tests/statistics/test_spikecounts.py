import numpy as np
import pytest

from miv.core.datatype.spikestamps import Spikestamps
from miv.statistics import decay_spike_counts, spike_counts_with_kernel


@pytest.fixture
def spiketrain():
    return np.array([1.5, 2.7, 3.1, 4.8, 5.2, 6.1])


@pytest.fixture
def probe_times():
    return np.linspace(0, 10, 5)


def test_no_spike_counts():
    result = decay_spike_counts(np.array([]), np.arange(10))
    np.testing.assert_allclose(np.zeros(10), result)


def test_empty_probe_times():
    """Test edge case where probe_times is empty (n_probe == 0)"""
    spiketrain = np.array([1.5, 2.7, 3.1])
    probe_times = np.array([])
    result = decay_spike_counts(spiketrain, probe_times)
    assert isinstance(result, np.ndarray)
    assert result.shape == (0,)
    assert result.dtype == np.float64
    np.testing.assert_allclose(np.array([]), result)


def test_decay_spike_counts(spiketrain, probe_times):
    result = decay_spike_counts(spiketrain, probe_times)
    print(repr(result))

    # Expected values calculated with hardcoded amplitude=2.0, decay_rate=5
    # The function uses: amplitude * exp(-decay_rate * x) * (decay_rate^2) * x
    # Using actual output values with appropriate tolerance for floating point precision
    expected = np.array(
        [0.000000e00, 3.368973e-01, 3.687070e00, 6.518185e-02, 6.730513e-07]
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == probe_times.shape
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_spike_counts_with_kernel(spiketrain, probe_times):
    def heaviside(x):
        return (x >= 0).astype(int)

    def exponential(x):
        return np.exp(-x)

    kernels = [heaviside, exponential]
    np.set_printoptions(precision=16)
    expected = [
        np.array([0, 1, 4, 6, 6]),
        np.array(
            [
                0.0,
                0.3678794411714423,
                1.098755599445739,
                0.4370471595329147,
                0.0358750154888382,
            ]
        ),
    ]

    for idx, kernel in enumerate(kernels):
        result = spike_counts_with_kernel(spiketrain, probe_times, kernel)
        assert isinstance(result, np.ndarray)
        assert result.shape == probe_times.shape
        np.testing.assert_allclose(result, expected[idx])


def test_spike_counts_with_kernel_empty_probe_times():
    """Test edge case where probe_times is empty for spike_counts_with_kernel"""
    spiketrain = np.array([1.5, 2.7, 3.1])
    probe_times = np.array([])

    def exponential(x):
        return np.exp(-x)

    result = spike_counts_with_kernel(spiketrain, probe_times, exponential)
    assert isinstance(result, np.ndarray)
    assert result.shape == (0,)
    np.testing.assert_allclose(np.array([]), result)
