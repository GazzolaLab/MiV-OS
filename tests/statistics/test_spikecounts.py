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
    result = decay_spike_counts(
        [], np.arange(10), amplitude=1.0, decay_rate=5, batchsize=256
    )
    np.testing.assert_allclose(np.zeros(10), result)


def test_decay_spike_counts(spiketrain, probe_times):
    result = decay_spike_counts(
        spiketrain, probe_times, amplitude=0.2, decay_rate=5, batchsize=256
    )
    print(repr(result))

    expected = np.array(
        [0.0e00, 6.73794700e-03, 3.67964448e-01, 9.23383335e-04, 3.44112943e-09]
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == probe_times.shape
    np.testing.assert_allclose(result, expected)


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
