import numpy as np
import pytest

from miv.coding.temporal.spiker import BensSpikerAlgorithm


def test_bsa_finite_impulse_response():
    # Create an instance of the BensSpikerAlgorithm class
    bsa = BensSpikerAlgorithm(
        sampling_rate=1000,
        threshold=1.0,
        fir_filter_length=3,
        fir_cutoff=0.5,
        normalize=False,
    )

    # Generate some test data
    data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    # Call the _finite_impulse_response method
    filter_values, filter_length = bsa._finite_impulse_response(data)

    # Assert that the returned filter values and filter length are as expected
    np.testing.assert_allclose(filter_values, [0.046221, 0.907557, 0.046221], rtol=2e-5)
    assert filter_length == 3


def test_bsa_init():
    # Create an instance of the BensSpikerAlgorithm class with default parameters
    bsa = BensSpikerAlgorithm(sampling_rate=1000)

    # Assert that the object's attributes have the correct values
    assert bsa.sampling_rate == 1000
    assert bsa.threshold == 1.0
    assert bsa.fir_filter_length == 2
    assert bsa.fir_cutoff == 0.8
    assert bsa.data_normalize is False

    # Create an instance of the BensSpikerAlgorithm class with non-default parameters
    bsa = BensSpikerAlgorithm(
        sampling_rate=500,
        threshold=2.0,
        fir_filter_length=3,
        fir_cutoff=0.5,
        normalize=True,
    )

    # Assert that the object's attributes have the correct values
    assert bsa.sampling_rate == 500
    assert bsa.threshold == 2.0
    assert bsa.fir_filter_length == 3
    assert bsa.fir_cutoff == 0.5
    assert bsa.data_normalize is True


def test_bsa_call():
    # Create an instance of the BensSpikerAlgorithm class with default parameters
    bsa = BensSpikerAlgorithm(sampling_rate=1000)

    # Generate some test data
    data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    # Call the __call__ method
    spiketrains, timestamps = bsa(data, time_offset=0)

    # Assert that the returned spiketrains and timestamps are as expected
    np.testing.assert_allclose(np.array([2, 5]), spiketrains.shape)
    np.testing.assert_allclose(np.array([0, 0.001]), timestamps)


def test_call_with_spikes():
    # Create an instance of the BensSpikerAlgorithm class with a threshold of 5
    bsa = BensSpikerAlgorithm(sampling_rate=1000, threshold=5)

    # Generate some test data with two spikes
    data = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
    )

    # Call the __call__ method
    spiketrains, timestamps = bsa(data, time_offset=0)

    # Assert that the returned spiketrains and timestamps are as expected
    assert spiketrains.shape == (2, 10)
    np.testing.assert_allclose(timestamps, np.array([0, 0.001]))
