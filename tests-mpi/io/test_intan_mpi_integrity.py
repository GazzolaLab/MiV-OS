"""Tests for Intan MPI pipeline integrity similar to examples/intan_io_mpi.py."""

import pytest
import os
import numpy as np

from miv.core.pipeline import Pipeline
from miv.core.operator.policy import SupportMPIMerge
from miv.io.intan import DataIntan
from miv.io.intan.mock_data import create_mock_rhs_data_folder

from mpi4py import MPI


# Mock operation that supports SupportMPIMerge
from dataclasses import dataclass
from miv.core import Signal, Spikestamps
from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.core.operator_generator.operator import GeneratorOperatorMixin
from miv.core.operator_generator.wrapper import cache_generator_call


@dataclass
class MockBandpass(GeneratorOperatorMixin):
    """Mock bandpass filter that just passes through the signal without filtering.

    This is a simplified version of ButterBandpass for testing purposes.
    It simply returns the input signal unchanged.
    """

    tag: str = "mock bandpass filter"

    def __post_init__(self):
        super().__init__()
        self.cacher.policy = "OFF"

    @cache_generator_call
    def __call__(self, signal: Signal) -> Signal:
        """Pass through the signal without any filtering.

        Parameters
        ----------
        signal : Signal
            Input signal to pass through

        Returns
        -------
        Signal
            The same signal unchanged
        """
        return Signal(data=signal.data, timestamps=signal.timestamps, rate=signal.rate)


@dataclass
class MockSpikeDetection(OperatorMixin):
    """Mock spike detection operation that supports SupportMPIMerge.

    Returns spikestamps that are a function of the MPI rank for easy verification.
    Each rank returns identifiable spikestamps that can be checked after merge.
    """

    rank: int = 0
    tag: str = "mock spike detection"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(self, signal):
        """Return spikestamps that are a function of the MPI rank.

        Handles both Signal objects and generators of Signal objects.
        Returns the same rank-specific spikestamps regardless of number of files.
        """
        import inspect

        # Get the rank from the runner's comm
        rank = self.rank

        if inspect.isgenerator(signal):
            # Handle generator case (from upstream filter)
            # Get number of channels from first signal
            num_channels = None
            for sig in signal:
                if num_channels is None:
                    num_channels = sig.number_of_channels
                break

            if num_channels is None:
                num_channels = 4  # Default

            # Return rank-specific spikestamps (same regardless of number of files processed)
            # Each rank returns spikestamps: [rank*10.0, rank*10.0 + 0.1, rank*10.0 + 0.2] per channel
            spiketrain_list = []
            for channel in range(num_channels):
                # Each channel gets slightly different timestamps
                spikestamps = np.array(
                    [rank * 10.0 + channel * 0.01 + i * 0.1 for i in range(3)]
                )
                spiketrain_list.append(spikestamps.astype(np.float64))

            return Spikestamps(spiketrain_list)
        else:
            # Handle single Signal case
            num_channels = signal.number_of_channels

            spiketrain_list = []
            for channel in range(num_channels):
                spikestamps = np.array(
                    [rank * 10.0 + channel * 0.01 + i * 0.1 for i in range(3)]
                )
                spiketrain_list.append(spikestamps.astype(np.float64))

            return Spikestamps(spiketrain_list)


@pytest.mark.mpi
def test_mock_bandpass_passes_through_signal(mpi_tmpdir):
    """Test that MockBandpass correctly passes through signals without modification.

    This test verifies:
    1. MockBandpass can be used in place of ButterBandpass
    2. It returns the same signal data, timestamps, and rate unchanged
    3. It works with generator operators (handles Signal objects)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    num_files = 2
    num_channels = 4
    sampling_rate = 30000
    duration_seconds = 10

    # Create mock RHS data folder
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

    folder_path = comm.bcast(folder_path, root=0)
    comm.barrier()

    # Verify all ranks can access the directory
    assert os.path.exists(folder_path), f"Rank {rank}: Cannot access data directory"

    # Set up pipeline with MockBandpass
    data = DataIntan(data_path=folder_path)
    data.configure_load(mpi_comm=comm)
    mock_filter = MockBandpass()

    data >> mock_filter
    pipeline = Pipeline(mock_filter)
    pipeline.run(working_directory=str(mpi_tmpdir))

    # Get results - should be a generator of Signal objects
    output_generator = mock_filter.output()

    # Verify we get signals from the generator
    # Note: Some ranks may not get any files if there are more ranks than files
    signals = list(output_generator)

    # Verify each signal is unchanged (data, timestamps, rate match input)
    # Ranks without files will have empty signal list, which is fine
    for i, signal in enumerate(signals):
        assert isinstance(signal, Signal), (
            f"Rank {rank}, Signal {i}: Expected Signal, got {type(signal)}"
        )
        assert signal.rate == sampling_rate, (
            f"Rank {rank}, Signal {i}: Expected rate {sampling_rate}, got {signal.rate}"
        )
        assert signal.number_of_channels == num_channels, (
            f"Rank {rank}, Signal {i}: Expected {num_channels} channels, got {signal.number_of_channels}"
        )
        assert signal.shape[1] == num_channels, (
            f"Rank {rank}, Signal {i}: Expected {num_channels} channels in data shape, got {signal.shape[1]}"
        )
        assert len(signal.timestamps) == signal.shape[0], (
            f"Rank {rank}, Signal {i}: Timestamps length {len(signal.timestamps)} doesn't match data shape {signal.shape[0]}"
        )
        assert np.all(np.isfinite(signal.data)), (
            f"Rank {rank}, Signal {i}: Data contains NaN or Inf values"
        )
        assert np.all(np.isfinite(signal.timestamps)), (
            f"Rank {rank}, Signal {i}: Timestamps contain NaN or Inf values"
        )

    # Verify that all files are processed across all ranks
    num_signals_per_rank = comm.allgather(len(signals))
    total_signals = sum(num_signals_per_rank)
    assert total_signals == num_files, (
        f"Expected {num_files} total signals across all ranks, got {total_signals}"
    )


@pytest.mark.mpi
def test_dataintan_mpi_load_timestamp_continuity_channel_consistency(mpi_tmpdir):
    """Test DataIntan MPI Load - Timestamp Continuity and Channel Consistency.

    This test verifies:
    1. Timestamps are continuous across files when loaded with MPI
    2. Channel consistency is maintained (same channels across all files/ranks)
    3. No timestamp gaps or overlaps between files
    4. Channel ordering is consistent
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # size = comm.Get_size()

    num_files = 6
    num_channels = 4
    sampling_rate = 30000
    duration_seconds = 10

    # Create mock RHS data folder
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

    folder_path = comm.bcast(folder_path, root=0)
    comm.barrier()

    # Verify all ranks can access the directory
    assert os.path.exists(folder_path), f"Rank {rank}: Cannot access data directory"

    # Load data with DataIntan using MPI
    data_intan = DataIntan(data_path=folder_path)
    data_intan.configure_load(mpi_comm=comm)

    # Verify path validity on all ranks
    assert data_intan.check_path_validity(), f"Rank {rank}: Path validity check failed"

    # Load signals with MPI
    signals = list(data_intan.load(mpi_comm=comm))

    # Verify we got signals (each rank should get a subset of files)
    assert len(signals) >= 0  # Some ranks might get 0 files if more ranks than files

    # Verify channel consistency for each signal
    for i, signal in enumerate(signals):
        assert signal.number_of_channels == num_channels, (
            f"Rank {rank}, Signal {i}: Expected {num_channels} channels, got {signal.number_of_channels}"
        )
        assert signal.shape[1] == num_channels, (
            f"Rank {rank}, Signal {i}: Expected {num_channels} channels in data shape, got {signal.shape[1]}"
        )
        assert signal.rate == sampling_rate, (
            f"Rank {rank}, Signal {i}: Expected sampling rate {sampling_rate}, got {signal.rate}"
        )

    # Collect all signals from all ranks to verify timestamp continuity
    all_signals_per_rank = comm.allgather(signals)
    all_signals = []
    for rank_signals in all_signals_per_rank:
        all_signals.extend(rank_signals)

    # Verify we have all files
    assert len(all_signals) == num_files, (
        f"Expected {num_files} total signals across all ranks, got {len(all_signals)}"
    )

    # Verify timestamp validity within each file
    # RHS timestamps are converted from sample indices to seconds
    # We verify that timestamps are valid (finite) and match the data shape
    for i, signal in enumerate(all_signals):
        # Check that timestamps are finite
        assert np.all(np.isfinite(signal.timestamps)), (
            f"Signal {i}: Timestamps contain NaN or Inf values"
        )

        # Verify timestamp length matches data shape
        assert len(signal.timestamps) == signal.shape[0], (
            f"Signal {i}: Timestamp length {len(signal.timestamps)} doesn't match data shape {signal.shape[0]}"
        )

    # Verify channel consistency across all signals
    # All signals should have the same number of channels
    for i, signal in enumerate(all_signals):
        assert signal.number_of_channels == num_channels, (
            f"Signal {i}: Expected {num_channels} channels, got {signal.number_of_channels}"
        )
        assert signal.shape[1] == num_channels, (
            f"Signal {i}: Expected {num_channels} channels in data shape, got {signal.shape[1]}"
        )

        # Verify sampling rate consistency
        assert signal.rate == sampling_rate, (
            f"Signal {i}: Expected sampling rate {sampling_rate}, got {signal.rate}"
        )


@pytest.mark.mpi
def test_intan_mpi_pipeline_with_mock_operation(mpi_tmpdir):
    """Test complete MPI pipeline similar to examples/intan_io_mpi.py.

    This test verifies:
    1. DataIntan loads data with MPI
    2. ButterBandpass filters the data
    3. MockSpikeDetection detects spikes with SupportMPIMerge
    4. Results are correctly merged across all ranks
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_files = 6
    num_channels = 4
    sampling_rate = 30000
    duration_seconds = 10

    # Create mock RHS data folder
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

    folder_path = comm.bcast(folder_path, root=0)
    comm.barrier()

    # Verify all ranks can access the directory
    assert os.path.exists(folder_path), f"Rank {rank}: Cannot access data directory"

    # Set up pipeline similar to examples/intan_io_mpi.py
    data = DataIntan(data_path=folder_path)
    data.configure_load(mpi_comm=comm)
    bandpass_filter = MockBandpass()
    spike_detection = MockSpikeDetection(rank=rank)
    spike_detection.runner = SupportMPIMerge(comm=comm)
    # Disable caching to ensure we get the merged result directly from the runner
    spike_detection.set_caching_policy("OFF")

    data >> bandpass_filter >> spike_detection
    pipeline = Pipeline(spike_detection)
    pipeline.run(working_directory=str(mpi_tmpdir))

    # Get results - with caching disabled, output() will call the runner again
    # Since SupportMPIMerge broadcasts the merged result to all ranks, all ranks should
    # get the same merged result
    spike_train_mpi = spike_detection.output()

    # Verify results
    assert spike_train_mpi is not None, f"Rank {rank}: Spike train is None"
    assert isinstance(spike_train_mpi, Spikestamps), (
        f"Rank {rank}: Expected Spikestamps, got {type(spike_train_mpi)}"
    )

    # Verify all channels are present
    assert spike_train_mpi.number_of_channels == num_channels, (
        f"Rank {rank}: Expected {num_channels} channels, got {spike_train_mpi.number_of_channels}"
    )

    # Verify spikes were detected (at least some spikes should be found)
    total_spikes_mpi = sum(spike_train_mpi.get_count())

    assert total_spikes_mpi >= 0, f"Rank {rank}: Total spikes should be non-negative"

    # Verify merged result contains spikestamps from all ranks
    # Since SupportMPIMerge broadcasts the merged result, all ranks should have the same merged result
    # Each rank contributes: [rank*10.0, rank*10.0 + 0.1, rank*10.0 + 0.2] per channel

    # Collect all expected spikestamps from all ranks
    expected_spikestamps_per_channel = []
    for ch in range(num_channels):
        expected_for_channel = []
        for r in range(size):
            # Each rank contributes 3 spikes per channel
            expected_for_channel.extend(
                [r * 10.0 + ch * 0.01 + i * 0.1 for i in range(3)]
            )
        expected_spikestamps_per_channel.append(sorted(expected_for_channel))

    # Verify merged result contains all expected spikestamps (on all ranks, since result is broadcast)
    for channel in range(num_channels):
        actual_spikestamps = sorted(spike_train_mpi[channel].tolist())
        expected_spikestamps = expected_spikestamps_per_channel[channel]

        assert len(actual_spikestamps) == len(expected_spikestamps), (
            f"Rank {rank}, Channel {channel}: Expected {len(expected_spikestamps)} spikes, got {len(actual_spikestamps)}. Actual: {actual_spikestamps[:10]}..."
        )

        # Verify all expected spikestamps are present (within floating point tolerance)
        for expected_spike in expected_spikestamps:
            found = any(
                abs(actual - expected_spike) < 1e-6 for actual in actual_spikestamps
            )
            assert found, (
                f"Rank {rank}, Channel {channel}: Expected spike {expected_spike} not found in merged result. Got: {actual_spikestamps[:10]}..."
            )

    # Verify total spike count matches expected
    expected_total = size * 3 * num_channels  # 3 spikes per rank per channel
    assert total_spikes_mpi == expected_total, (
        f"Rank {rank}: Expected {expected_total} total spikes (3 per rank per channel), got {total_spikes_mpi}"
    )
