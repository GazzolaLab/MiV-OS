"""Tests for operator policy with MPI support."""

import pytest

from miv.core.operator.policy import VanillaRunner

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
