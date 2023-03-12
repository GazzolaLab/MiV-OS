import numpy as np
import pytest

from miv.signal.spike import ThresholdCutoff


@pytest.fixture
def threshold_cutoff():
    return ThresholdCutoff()


def test_compute_spike_threshold(threshold_cutoff):
    signal = np.array([1, 2, 3, 4, 5])
    cutoff = 3.0
    use_mad = True
    expected_output = -np.median(np.absolute(signal)) / 0.6745 * cutoff
    assert (
        threshold_cutoff._compute_spike_threshold(signal, cutoff, use_mad)
        == expected_output
    )


def test_detect_threshold_crossings(threshold_cutoff):
    signal = np.array([1, 2, 3, 4, 5])
    fs = 1.0
    threshold = 2.0
    dead_time = 0.5
    expected_output = np.array([1])
    np.testing.assert_array_equal(
        threshold_cutoff._detect_threshold_crossings(signal, fs, threshold, dead_time),
        expected_output,
    )


def test_align_to_minimum(threshold_cutoff):
    signal = np.array([1, 2, 3, 2, 1])
    fs = 1.0
    threshold_crossings = np.array([1, 3])
    search_range = 1.0
    expected_output = np.array([1, 3])
    np.testing.assert_array_equal(
        threshold_cutoff._align_to_minimum(
            signal, fs, threshold_crossings, search_range
        ),
        expected_output,
    )
