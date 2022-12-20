import numpy as np

from miv.io.binary import load_ttl_event
from tests.io.mock_data import test_TTL_data


def test_load_ttl_event(test_TTL_data):
    (
        tmp_path,
        expected_states,
        expected_full_words,
        expected_timestamps,
        expected_sampling_rate,
        expected_initial_state,
        expected_sample_numbers,
    ) = test_TTL_data

    # Test the load_ttl_event function with return_sample_numbers=False
    states, full_words, timestamps, sampling_rate, initial_state = load_ttl_event(
        tmp_path
    )
    np.testing.assert_allclose(states, expected_states)
    np.testing.assert_allclose(full_words, expected_full_words)
    np.testing.assert_allclose(timestamps, expected_timestamps)
    np.testing.assert_allclose(sampling_rate, expected_sampling_rate)
    np.testing.assert_allclose(initial_state, expected_initial_state)

    # Test the load_ttl_event function with return_sample_numbers=True
    (
        states,
        full_words,
        timestamps,
        sampling_rate,
        initial_state,
        sample_numbers,
    ) = load_ttl_event(tmp_path, return_sample_numbers=True)
    np.testing.assert_allclose(states, expected_states)
    np.testing.assert_allclose(full_words, expected_full_words)
    np.testing.assert_allclose(timestamps, expected_timestamps)
    np.testing.assert_allclose(sampling_rate, expected_sampling_rate)
    np.testing.assert_allclose(initial_state, expected_initial_state)
    np.testing.assert_allclose(sample_numbers, expected_sample_numbers)
