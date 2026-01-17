"""Tests for reading mock Intan RHS data with MPI."""

import pytest
import os
import numpy as np

from miv.io.intan import DataIntan
from miv.io.intan.mock_data import create_mock_rhs_data_folder

from mpi4py import MPI


@pytest.mark.mpi
@pytest.mark.parametrize("num_files", [1, 3, 6])
def test_read_mock_rhs_data_folder_with_dataintan_mpi(mpi_tmpdir, num_files):
    """Test reading mock RHS data folder with DataIntan using MPI communicator.

    Verify that data is loaded correctly in each rank.
    """
    if MPI is None:
        pytest.skip("mpi4py not available")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # size = comm.Get_size()

    num_channels = 4
    sampling_rate = 30000
    duration_seconds = 10

    # Create mock RHS data folder (only on rank 0 to avoid race conditions)
    # This is a setup step - we're testing MPI data loading, not file creation
    if rank == 0:
        folder_path = create_mock_rhs_data_folder(
            output_dir=mpi_tmpdir,
            num_files=num_files,
            num_channels=num_channels,
            sampling_rate=sampling_rate,
            duration_seconds=duration_seconds,
        )
    else:
        folder_path = None

    # Broadcast folder path to all ranks
    folder_path = comm.bcast(folder_path, root=0)

    # Ensure all ranks wait for rank 0 to finish creating files
    comm.barrier()

    # Verify all ranks can access the data directory
    assert os.path.exists(folder_path), (
        f"Rank {rank}: Cannot access data directory {folder_path}"
    )
    assert os.path.exists(os.path.join(folder_path, "settings.xml")), (
        f"Rank {rank}: Cannot access settings.xml"
    )

    # Load data with DataIntan using MPI
    data_intan = DataIntan(data_path=folder_path)

    # Verify path validity on all ranks
    assert data_intan.check_path_validity(), f"Rank {rank}: Path validity check failed"

    # Load signals with MPI
    signals = list(data_intan.load(mpi_comm=comm))

    # Verify we got signals (each rank should get a subset of files)
    # The number of signals per rank depends on how files are distributed
    assert len(signals) >= 0  # Some ranks might get 0 files if more ranks than files

    # Verify each signal loaded by this rank
    expected_samples_per_file = int(sampling_rate * duration_seconds)
    expected_samples_rounded = ((expected_samples_per_file + 127) // 128) * 128

    for i, signal in enumerate(signals):
        # Verify signal shape: [samples, channels]
        assert signal.shape[0] == expected_samples_rounded, (
            f"Rank {rank}, File {i}: expected {expected_samples_rounded} samples, got {signal.shape[0]}"
        )
        assert signal.shape[1] == num_channels, (
            f"Rank {rank}, File {i}: expected {num_channels} channels, got {signal.shape[1]}"
        )

        # Verify sampling rate
        assert signal.rate == sampling_rate, (
            f"Rank {rank}, File {i}: expected sampling rate {sampling_rate}, got {signal.rate}"
        )

        # Verify timestamps shape matches data shape
        assert len(signal.timestamps) == signal.shape[0], (
            f"Rank {rank}, File {i}: timestamps length {len(signal.timestamps)} doesn't match data shape {signal.shape[0]}"
        )

        # Verify timestamps are numeric (not NaN or Inf)
        assert np.all(np.isfinite(signal.timestamps)), (
            f"Rank {rank}, File {i}: timestamps contain NaN or Inf values"
        )

        # Verify data is numeric (not NaN or Inf)
        assert np.all(np.isfinite(signal.data)), (
            f"Rank {rank}, File {i}: data contains NaN or Inf values"
        )

    # Verify that all files are processed across all ranks
    # Collect number of signals from each rank
    num_signals_per_rank = comm.allgather(len(signals))
    total_signals = sum(num_signals_per_rank)

    # All files should be processed exactly once across all ranks
    assert total_signals == num_files, (
        f"Expected {num_files} total signals across all ranks, got {total_signals}"
    )
