import json
import os

import numpy as np
import pytest


@pytest.fixture
def test_TTL_data(tmp_path):
    # Set up test data
    folder = tmp_path / "ttl_events"
    expected_states = [1, 0, 1, 0, 1]
    expected_full_words = [0, 1, 2, 3, 4]
    expected_timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
    expected_sampling_rate = 1000.0
    expected_initial_state = 0
    expected_sample_numbers = [0, 1000, 2000, 3000, 4000]

    # Create the test data files
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "events"), exist_ok=True)
    os.makedirs(os.path.join(folder, "events", "TTL_1"), exist_ok=True)
    np.save(os.path.join(folder, "events", "TTL_1", "states.npy"), expected_states)
    np.save(
        os.path.join(folder, "events", "TTL_1", "full_words.npy"), expected_full_words
    )
    np.save(
        os.path.join(folder, "events", "TTL_1", "timestamps.npy"), expected_timestamps
    )
    np.save(
        os.path.join(folder, "events", "TTL_1", "sample_numbers.npy"),
        expected_sample_numbers,
    )
    info = {
        "GUI version": "0.6.0",
        "events": [
            {
                "channel_name": "TTL Input",
                "folder_name": "TTL_1",
                "sample_rate": expected_sampling_rate,
                "initial_state": expected_initial_state,
            }
        ],
    }
    with open(os.path.join(folder, "structure.oebin"), "w") as f:
        json.dump(info, f)

    return (
        folder,
        expected_states,
        expected_full_words,
        expected_timestamps,
        expected_sampling_rate,
        expected_initial_state,
        expected_sample_numbers,
    )
