import pytest
import os
import numpy as np
from pathlib import Path

from miv.io.intan import DataIntan
from miv.io.intan.mock_data import create_mock_rhs_data_folder


def test_create_mock_rhs_data_folder(tmpdir):
    """Test creating a folder with mock RHS data."""
    num_files = 3
    num_channels = 4
    sampling_rate = 30000
    duration_seconds = 10

    folder_path = create_mock_rhs_data_folder(
        output_dir=str(tmpdir),
        num_files=num_files,
        num_channels=num_channels,
        sampling_rate=sampling_rate,
        duration_seconds=duration_seconds,
    )

    # Verify folder was created
    assert os.path.exists(folder_path)

    # Verify settings.xml exists
    settings_path = os.path.join(folder_path, "settings.xml")
    assert os.path.exists(settings_path)

    # Verify RHS files exist
    rhs_files = sorted(Path(folder_path).glob("*.rhs"))
    assert len(rhs_files) == num_files

    # Verify each file exists
    for rhs_file in rhs_files:
        assert os.path.exists(rhs_file)
        assert rhs_file.suffix == ".rhs"


def test_read_mock_rhs_data_folder_with_dataintan(tmpdir):
    """Test reading mock RHS data folder with DataIntan and verify data is loaded correctly."""
    num_files = 3
    num_channels = 4
    sampling_rate = 30000
    duration_seconds = 10

    # Create mock RHS data folder
    folder_path = create_mock_rhs_data_folder(
        output_dir=str(tmpdir),
        num_files=num_files,
        num_channels=num_channels,
        sampling_rate=sampling_rate,
        duration_seconds=duration_seconds,
    )

    # Load data with DataIntan
    data_intan = DataIntan(data_path=folder_path)

    # Verify path validity
    assert data_intan.check_path_validity()

    # Load signals
    signals = list(data_intan.load())

    # Verify we got the correct number of signals (one per file)
    assert len(signals) == num_files

    # Verify each signal
    # RHS files must be in 128-sample blocks, so we round up
    expected_samples_per_file = int(sampling_rate * duration_seconds)
    expected_samples_rounded = ((expected_samples_per_file + 127) // 128) * 128
    for i, signal in enumerate(signals):
        # Verify signal shape: [samples, channels]
        # Files are padded to 128-sample boundaries
        assert signal.shape[0] == expected_samples_rounded, (
            f"File {i}: expected {expected_samples_rounded} samples (rounded from {expected_samples_per_file}), got {signal.shape[0]}"
        )
        assert signal.shape[1] == num_channels, (
            f"File {i}: expected {num_channels} channels, got {signal.shape[1]}"
        )

        # Verify sampling rate
        assert signal.rate == sampling_rate, (
            f"File {i}: expected sampling rate {sampling_rate}, got {signal.rate}"
        )

        # Verify timestamps shape matches data shape
        assert len(signal.timestamps) == signal.shape[0], (
            f"File {i}: timestamps length {len(signal.timestamps)} doesn't match data shape {signal.shape[0]}"
        )

        # Verify timestamps are numeric (not NaN or Inf)
        # Note: Timestamp values may vary based on how RHS reader interprets them
        assert np.all(np.isfinite(signal.timestamps)), (
            f"File {i}: timestamps contain NaN or Inf values"
        )

        # Verify data is numeric (not NaN or Inf)
        assert np.all(np.isfinite(signal.data)), (
            f"File {i}: data contains NaN or Inf values"
        )
