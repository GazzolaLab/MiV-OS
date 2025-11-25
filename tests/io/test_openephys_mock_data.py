import os
import pytest
import numpy as np
from pathlib import Path

from miv.io.openephys.data import Data, DataManager
from miv.io.openephys.mock_data import create_recording_data, create_dataset


@pytest.mark.parametrize("duration_seconds", [10.0, 5.0])
def test_create_recording_data_creates_compatible_data(tmpdir, duration_seconds: float):
    """Test that create_recording_data creates data loadable with miv.io.openephys.Data."""
    num_channels = 4
    sampling_rate = 30000.0

    data_path, expected_data, expected_timestamps, sampling_rate = (
        create_recording_data(
            output_dir=str(tmpdir),
            num_channels=num_channels,
            duration_seconds=duration_seconds,
            sampling_rate=sampling_rate,
        )
    )

    assert os.path.exists(data_path)
    data = Data(data_path)
    assert data.check_path_validity()

    assert data.number_of_channels == num_channels
    signals = list(data.load())
    assert len(signals) > 0

    # Verify first signal has correct shape and properties
    signal = signals[0]
    expected_samples = int(sampling_rate * duration_seconds)
    assert signal.shape[0] == expected_samples
    assert signal.shape[1] == num_channels
    assert signal.rate == sampling_rate
    assert len(signal.timestamps) == expected_samples
    np.testing.assert_allclose(signal.timestamps, expected_timestamps)
    np.testing.assert_allclose(signal.data, expected_data)


def test_create_dataset_creates_multiple_recordings_loadable_with_datamanager(tmpdir):
    """Test that create_dataset creates multiple recording data loadable with DataManager."""
    num_record_nodes = 2
    num_experiments = 3
    num_recordings_per_experiment = 2
    num_channels = 4
    sampling_rate = 30000.0
    duration_seconds = 5.0

    data_collection_path = create_dataset(
        output_dir=str(tmpdir),
        num_record_nodes=num_record_nodes,
        num_experiments=num_experiments,
        num_recordings_per_experiment=num_recordings_per_experiment,
        num_channels=num_channels,
        sampling_rate=sampling_rate,
        duration_seconds=duration_seconds,
    )

    # Verify path exists
    assert os.path.exists(data_collection_path)

    # Verify it can be loaded with DataManager
    data_manager = DataManager(data_collection_path)

    # Verify number of recordings
    expected_num_recordings = (
        num_record_nodes * num_experiments * num_recordings_per_experiment
    )
    assert len(data_manager) == expected_num_recordings

    # Verify each recording is valid and can be loaded
    for data in data_manager:
        assert data.check_path_validity()
        assert data.number_of_channels == num_channels
        signals = list(data.load())
        assert len(signals) > 0
        signal = signals[0]
        expected_samples = int(sampling_rate * duration_seconds)
        assert signal.shape[0] == expected_samples
        assert signal.shape[1] == num_channels
        assert signal.rate == sampling_rate
