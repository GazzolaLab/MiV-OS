"""Tests for operator policy with MPI support."""

import pytest
from unittest.mock import patch
import numpy as np

from miv.core.operator.policy import (
    VanillaRunner,
    SupportMPIMerge,
    SupportMPIWithoutBroadcast,
)
from miv.core import Signal, Spikestamps

from mpi4py import MPI


@pytest.mark.mpi
def test_vanilla_runner_gets_global_comm_when_not_provided():
    """
    If comm is not explicitly provided, VanillaRunner should get global comm from mpi4py.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Only test if we have multiple ranks
    if size == 1:
        pytest.skip("Test requires multiple MPI ranks")

    # Create VanillaRunner without providing comm
    vr = VanillaRunner()

    # Should have gotten the global comm
    assert vr.comm is not None, (
        f"Rank {rank}: comm should not be None when MPI is available"
    )
    assert vr.comm == comm, f"Rank {rank}: comm should be MPI.COMM_WORLD"
    assert vr.is_root == (rank == 0), (
        f"Rank {rank}: is_root should be True only for rank 0"
    )


@pytest.mark.mpi
def test_vanilla_runner_uses_provided_comm():
    """
    If comm is explicitly provided, VanillaRunner should use the provided comm.
    This is to support sub-group communication.
    """
    comm_world = MPI.COMM_WORLD
    rank_world = comm_world.Get_rank()
    size_world = comm_world.Get_size()

    # Only test if we have multiple ranks
    if size_world == 1:
        pytest.skip("Test requires multiple MPI ranks")

    # Create a sub-communicator for testing (split into two groups)
    # Group 0: ranks 0, 1, ... (if size >= 2)
    # Group 1: ranks 2, 3, ... (if size >= 4)
    color = 0 if rank_world < size_world // 2 else 1
    key = rank_world
    sub_comm = comm_world.Split(color, key)

    try:
        # Create VanillaRunner with the provided sub-communicator
        vr = VanillaRunner(comm=sub_comm)

        # Should use the provided comm, not COMM_WORLD
        assert vr.comm is not None, f"Rank {rank_world}: comm should not be None"
        assert vr.comm == sub_comm, (
            f"Rank {rank_world}: comm should be the provided sub_comm"
        )
        assert vr.comm != comm_world, (
            f"Rank {rank_world}: comm should not be COMM_WORLD"
        )

        # Verify is_root is set correctly for the sub-communicator
        sub_rank = sub_comm.Get_rank()
        assert vr.is_root == (sub_rank == 0), (
            f"Rank {rank_world}: is_root should be True only for rank 0 in sub_comm"
        )
    finally:
        # Clean up the sub-communicator
        sub_comm.Free()


@pytest.mark.mpi(min_size=2)
def test_vanilla_runner_executes_only_on_root_and_broadcasts():
    """
    During the execution __call__, function should be executed only on root rank,
    and result should be broadcasted to all ranks.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    vr = VanillaRunner()
    execution_count = 0

    def func():
        # This function should only execute on root rank
        nonlocal execution_count
        execution_count += 1
        return 42  # Return a fixed value to verify broadcast

    result = vr(func, inputs=None)
    all_counts = comm.allgather(execution_count)
    total_executions = sum(all_counts)

    # Function should execute exactly once (only on root rank 0)
    assert total_executions == 1, (
        f"Function should execute exactly once, but executed {total_executions} times. "
        f"Counts per rank: {all_counts}"
    )
    assert all_counts[0] == 1, "Function should execute on root rank 0"
    for i in range(1, size):
        assert all_counts[i] == 0, f"Function should not execute on non-root rank {i}"

    # Result should be the same on all ranks (broadcasted from root)
    expected_result = 42  # Value from root rank
    assert result == expected_result, (
        f"Rank {rank}: Result should be {expected_result} (broadcasted from root), "
        f"but got {result}"
    )


@pytest.mark.mpi(min_size=2)
def test_vanilla_runner_returns_same_result_on_all_ranks_after_broadcast():
    """
    VanillaRunner should return same result on all ranks after broadcast.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    _size = comm.Get_size()

    vr = VanillaRunner()

    def func(x, y):
        # Return a value that includes the input to verify it's from root
        return x * y + 100

    # Call with inputs
    result = vr(func, inputs=(5, 3))

    # Gather results from all ranks
    all_results = comm.allgather(result)

    # All ranks should have the same result (broadcasted from root)
    expected_result = 5 * 3 + 100  # 115, computed on root rank
    assert all(r == expected_result for r in all_results), (
        f"All ranks should have the same result {expected_result}, "
        f"but got: {all_results}"
    )

    # Verify each rank got the correct result
    assert result == expected_result, (
        f"Rank {rank}: Result should be {expected_result}, but got {result}"
    )


@pytest.mark.mpi(min_size=2)
def test_support_mpi_merge_executes_on_each_rank_gathers_and_broadcasts():
    """
    SupportMPIMerge.__call__() should execute the function on each rank,
    gather results, concatenate them, and broadcast the result to all ranks.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    runner = SupportMPIMerge()

    # Each rank returns different data based on its rank
    def func():
        return f"rank_{rank}_data"

    # Mock the concatenate function to track calls and preserve order
    with patch("miv.core.operator.policy.concatenate") as mock_concatenate:
        # Mock concatenate to return a combined result
        # The concatenate function should receive results in rank order
        def mock_concatenate_impl(values):
            # Verify we received results from all ranks
            assert len(values) == size, (
                f"Rank {rank}: concatenate should receive {size} values, "
                f"but got {len(values)}"
            )
            # Verify order: results should be in rank order (0, 1, 2, ...)
            for i, val in enumerate(values):
                expected = f"rank_{i}_data"
                assert val == expected, (
                    f"Rank {rank}: concatenate received value {val} at index {i}, "
                    f"but expected {expected}. Order should be preserved."
                )
            # Return a combined result
            return f"concatenated_{'_'.join(values)}"

        mock_concatenate.side_effect = mock_concatenate_impl

        # Execute the runner
        result = runner(func, inputs=None)

        # Verify concatenate was called only on root rank
        if runner.is_root():
            assert mock_concatenate.called, (
                f"Rank {rank}: concatenate should be called on root rank"
            )
            # Verify concatenate was called with gathered results
            call_args = mock_concatenate.call_args[0][0]
            assert len(call_args) == size, (
                f"Rank {rank}: concatenate should be called with {size} values"
            )
        else:
            # Non-root ranks should not call concatenate directly
            # (it's called on root after gather)
            pass

    # All ranks should receive the same broadcasted result
    all_results = comm.allgather(result)
    expected_result = (
        f"concatenated_{'_'.join([f'rank_{i}_data' for i in range(size)])}"
    )
    assert all(r == expected_result for r in all_results), (
        f"All ranks should have the same result {expected_result}, "
        f"but got: {all_results}"
    )
    assert result == expected_result, (
        f"Rank {rank}: Result should be {expected_result}, but got {result}"
    )


@pytest.mark.mpi(min_size=2)
def test_support_mpi_merge_handles_signal():
    """
    SupportMPIMerge.__call__() should handle Signal data type,
    gathering and concatenating them correctly across all ranks.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    runner = SupportMPIMerge()

    # Each rank creates a portion of a signal
    # Rank 0: data[0:2], Rank 1: data[2:4], etc.
    def func_signal():
        samples_per_rank = 2
        start_idx = rank * samples_per_rank

        # Create signal data for this rank
        # Total will be size * samples_per_rank samples with 2 channels
        data = np.array(
            [
                [start_idx * 10 + i, start_idx * 10 + i + 1]
                for i in range(samples_per_rank)
            ]
        )
        timestamps = np.array([start_idx + i for i in range(samples_per_rank)])
        return Signal(data=data, timestamps=timestamps, rate=1000.0)

    result_signal = runner(func_signal, inputs=None)

    # Verify all ranks receive the same concatenated signal
    all_results_signal = comm.allgather(result_signal)
    assert all(isinstance(r, Signal) for r in all_results_signal), (
        "All ranks should receive Signal objects"
    )

    # Verify all ranks have the same result
    expected_shape = (size * 2, 2)  # size * samples_per_rank samples, 2 channels
    for r in all_results_signal:
        assert r.shape == expected_shape, (
            f"Rank {rank}: Signal shape should be {expected_shape}, but got {r.shape}"
        )
        assert r.number_of_channels == 2, (
            f"Rank {rank}: Signal should have 2 channels, but got {r.number_of_channels}"
        )

    # Verify the data is correctly concatenated (check first and last samples)
    if runner.is_root():
        # Check that timestamps are in order
        assert np.all(np.diff(result_signal.timestamps) >= 0), (
            "Timestamps should be in ascending order after concatenation"
        )
        # Check that data values match expected pattern
        assert result_signal.timestamps[0] == 0, "First timestamp should be 0"
        assert result_signal.timestamps[-1] == size * 2 - 1, (
            f"Last timestamp should be {size * 2 - 1}"
        )


@pytest.mark.mpi(min_size=2)
def test_support_mpi_merge_handles_spikestamps():
    """
    SupportMPIMerge.__call__() should handle Spikestamps data type,
    gathering and concatenating them correctly across all ranks.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    runner = SupportMPIMerge()

    # Each rank creates spikestamps for different channels
    # Rank 0: channel 0, Rank 1: channel 1, etc.
    def func_spikestamps():
        spikes = [rank * 10.0 + i for i in range(3)]  # 3 spikes per rank
        # Create Spikestamps with spikes on this rank's channel
        # We need to ensure all ranks have the same number of channels
        spikestamps = Spikestamps.empty()
        # Add empty channels for ranks before this one
        for _ in range(rank):
            spikestamps.append([])
        # Add spikes for this rank's channel
        spikestamps.append(spikes)
        # Add empty channels for ranks after this one
        for _ in range(rank + 1, size):
            spikestamps.append([])
        return spikestamps

    result_spikestamps = runner(func_spikestamps, inputs=None)

    # Verify all ranks receive the same concatenated spikestamps
    all_results_spikestamps = comm.allgather(result_spikestamps)
    assert all(isinstance(r, Spikestamps) for r in all_results_spikestamps), (
        "All ranks should receive Spikestamps objects"
    )

    # Verify all ranks have the same result
    expected_num_channels = size
    for r in all_results_spikestamps:
        assert r.number_of_channels == expected_num_channels, (
            f"Rank {rank}: Spikestamps should have {expected_num_channels} channels, "
            f"but got {r.number_of_channels}"
        )

    # Verify the spikes are correctly concatenated
    if runner.is_root():
        # Each channel should have 3 spikes
        for ch_idx in range(size):
            channel_spikes = result_spikestamps[ch_idx]
            assert len(channel_spikes) == 3, (
                f"Channel {ch_idx} should have 3 spikes, but got {len(channel_spikes)}"
            )
            # Verify spikes match expected values
            expected_spikes = [ch_idx * 10.0 + i for i in range(3)]
            np.testing.assert_array_equal(
                channel_spikes,
                expected_spikes,
                err_msg=f"Channel {ch_idx} spikes should be {expected_spikes}",
            )


@pytest.mark.mpi(min_size=2)
def test_support_mpi_without_broadcast_executes_gathers_and_concatenates():
    """
    SupportMPIWithoutBroadcast.__call__() should execute the function on each rank,
    gather results, concatenate them, but NOT broadcast.
    Only root rank should see the result, non-root ranks should return None.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    runner = SupportMPIWithoutBroadcast()

    # Each rank returns different data based on its rank
    def func():
        return f"rank_{rank}_data"

    # Mock the concatenate function to track calls and preserve order
    with patch("miv.core.operator.policy.concatenate") as mock_concatenate:
        # Mock concatenate to return a combined result
        # The concatenate function should receive results in rank order
        def mock_concatenate_impl(values):
            # Verify we received results from all ranks
            assert len(values) == size, (
                f"Rank {rank}: concatenate should receive {size} values, "
                f"but got {len(values)}"
            )
            # Verify order: results should be in rank order (0, 1, 2, ...)
            for i, val in enumerate(values):
                expected = f"rank_{i}_data"
                assert val == expected, (
                    f"Rank {rank}: concatenate received value {val} at index {i}, "
                    f"but expected {expected}. Order should be preserved."
                )
            # Return a combined result
            return f"concatenated_{'_'.join(values)}"

        mock_concatenate.side_effect = mock_concatenate_impl

        # Execute the runner
        result = runner(func, inputs=None)

        # Verify concatenate was called only on root rank
        if runner.is_root():
            assert mock_concatenate.called, (
                f"Rank {rank}: concatenate should be called on root rank"
            )
            # Verify concatenate was called with gathered results
            call_args = mock_concatenate.call_args[0][0]
            assert len(call_args) == size, (
                f"Rank {rank}: concatenate should be called with {size} values"
            )
            # Root rank should receive the concatenated result
            expected_result = (
                f"concatenated_{'_'.join([f'rank_{i}_data' for i in range(size)])}"
            )
            assert result == expected_result, (
                f"Rank {rank}: Root rank should receive {expected_result}, "
                f"but got {result}"
            )
        else:
            # Non-root ranks should NOT call concatenate
            # Non-root ranks should return None (not broadcasted)
            assert result is None, (
                f"Rank {rank}: Non-root rank should return None, but got {result}"
            )

    # Verify results are NOT broadcasted - gather all results to verify
    all_results = comm.allgather(result)
    # Root rank should have the concatenated result
    expected_result = (
        f"concatenated_{'_'.join([f'rank_{i}_data' for i in range(size)])}"
    )
    assert all_results[0] == expected_result, (
        f"Root rank should have {expected_result}, but got {all_results[0]}"
    )
    # Non-root ranks should have None
    for i in range(1, size):
        assert all_results[i] is None, (
            f"Rank {i} should have None (not broadcasted), but got {all_results[i]}"
        )


@pytest.mark.mpi(min_size=2)
def test_support_mpi_without_broadcast_handles_signal():
    """
    SupportMPIWithoutBroadcast.__call__() should handle Signal data type,
    gathering and concatenating them correctly, but only root rank should see the result.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    runner = SupportMPIWithoutBroadcast()

    # Each rank creates a portion of a signal
    # Rank 0: data[0:2], Rank 1: data[2:4], etc.
    def func_signal():
        samples_per_rank = 2
        start_idx = rank * samples_per_rank

        # Create signal data for this rank
        # Total will be size * samples_per_rank samples with 2 channels
        data = np.array(
            [
                [start_idx * 10 + i, start_idx * 10 + i + 1]
                for i in range(samples_per_rank)
            ]
        )
        timestamps = np.array([start_idx + i for i in range(samples_per_rank)])
        return Signal(data=data, timestamps=timestamps, rate=1000.0)

    result_signal = runner(func_signal, inputs=None)

    # Verify root rank receives the concatenated signal, non-root ranks get None
    if runner.is_root():
        assert isinstance(result_signal, Signal), (
            f"Rank {rank}: Root rank should receive Signal object, but got {type(result_signal)}"
        )
        # Verify the concatenated signal shape
        expected_shape = (size * 2, 2)  # size * samples_per_rank samples, 2 channels
        assert result_signal.shape == expected_shape, (
            f"Rank {rank}: Signal shape should be {expected_shape}, but got {result_signal.shape}"
        )
        assert result_signal.number_of_channels == 2, (
            f"Rank {rank}: Signal should have 2 channels, but got {result_signal.number_of_channels}"
        )
        # Check that timestamps are in order
        assert np.all(np.diff(result_signal.timestamps) >= 0), (
            "Timestamps should be in ascending order after concatenation"
        )
        # Check that data values match expected pattern
        assert result_signal.timestamps[0] == 0, "First timestamp should be 0"
        assert result_signal.timestamps[-1] == size * 2 - 1, (
            f"Last timestamp should be {size * 2 - 1}"
        )
    else:
        # Non-root ranks should return None (not broadcasted)
        assert result_signal is None, (
            f"Rank {rank}: Non-root rank should return None, but got {result_signal}"
        )

    # Verify results are NOT broadcasted - gather all results to verify
    all_results_signal = comm.allgather(result_signal)
    # Root rank should have the Signal object
    assert isinstance(all_results_signal[0], Signal), (
        "Root rank should have Signal object"
    )
    # Non-root ranks should have None
    for i in range(1, size):
        assert all_results_signal[i] is None, (
            f"Rank {i} should have None (not broadcasted), but got {all_results_signal[i]}"
        )


@pytest.mark.mpi(min_size=2)
def test_support_mpi_without_broadcast_handles_spikestamps():
    """
    SupportMPIWithoutBroadcast.__call__() should handle Spikestamps data type,
    gathering and concatenating them correctly, but only root rank should see the result.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    runner = SupportMPIWithoutBroadcast()

    # Each rank creates spikestamps for different channels
    # Rank 0: channel 0, Rank 1: channel 1, etc.
    def func_spikestamps():
        spikes = [rank * 10.0 + i for i in range(3)]  # 3 spikes per rank
        # Create Spikestamps with spikes on this rank's channel
        # We need to ensure all ranks have the same number of channels
        spikestamps = Spikestamps.empty()
        # Add empty channels for ranks before this one
        for _ in range(rank):
            spikestamps.append([])
        # Add spikes for this rank's channel
        spikestamps.append(spikes)
        # Add empty channels for ranks after this one
        for _ in range(rank + 1, size):
            spikestamps.append([])
        return spikestamps

    result_spikestamps = runner(func_spikestamps, inputs=None)

    # Verify root rank receives the concatenated spikestamps, non-root ranks get None
    if runner.is_root():
        assert isinstance(result_spikestamps, Spikestamps), (
            f"Rank {rank}: Root rank should receive Spikestamps object, "
            f"but got {type(result_spikestamps)}"
        )
        # Verify the concatenated spikestamps
        expected_num_channels = size
        assert result_spikestamps.number_of_channels == expected_num_channels, (
            f"Rank {rank}: Spikestamps should have {expected_num_channels} channels, "
            f"but got {result_spikestamps.number_of_channels}"
        )
        # Each channel should have 3 spikes
        for ch_idx in range(size):
            channel_spikes = result_spikestamps[ch_idx]
            assert len(channel_spikes) == 3, (
                f"Channel {ch_idx} should have 3 spikes, but got {len(channel_spikes)}"
            )
            # Verify spikes match expected values
            expected_spikes = [ch_idx * 10.0 + i for i in range(3)]
            np.testing.assert_array_equal(
                channel_spikes,
                expected_spikes,
                err_msg=f"Channel {ch_idx} spikes should be {expected_spikes}",
            )
    else:
        # Non-root ranks should return None (not broadcasted)
        assert result_spikestamps is None, (
            f"Rank {rank}: Non-root rank should return None, but got {result_spikestamps}"
        )

    # Verify results are NOT broadcasted - gather all results to verify
    all_results_spikestamps = comm.allgather(result_spikestamps)
    # Root rank should have the Spikestamps object
    assert isinstance(all_results_spikestamps[0], Spikestamps), (
        "Root rank should have Spikestamps object"
    )
    # Non-root ranks should have None
    for i in range(1, size):
        assert all_results_spikestamps[i] is None, (
            f"Rank {i} should have None (not broadcasted), but got {all_results_spikestamps[i]}"
        )
