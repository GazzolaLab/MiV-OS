"""Tests for StrictMPIRunner policy with MPI support."""

import pytest

from miv.core.operator.policy import StrictMPIRunner

from mpi4py import MPI


@pytest.mark.mpi(min_size=2)
def test_mpi_comm():
    """
    Test StrictMPIRunner MPI communicator methods in a single test function.
    Tests get_rank(), get_size(), get_root(), and is_root() methods.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    runner = StrictMPIRunner(comm=comm)
    assert runner.get_rank() == rank, (
        f"Rank {rank}: get_rank() should return {rank}, but got {runner.get_rank()}"
    )
    assert runner.get_size() == size, (
        f"Rank {rank}: get_size() should return {size}, but got {runner.get_size()}"
    )
    assert runner.get_root() == 0, (
        f"Rank {rank}: get_root() should return 0 by default, but got {runner.get_root()}"
    )
    assert runner.is_root() == (rank == 0), (
        f"Rank {rank}: is_root() should return {rank == 0}, but got {runner.is_root()}"
    )

    # Test get_root() returns custom root when provided
    # Use a root that exists (e.g., rank 1 if size >= 2, otherwise 0)
    custom_root = 1 if size >= 2 else 0
    runner_custom = StrictMPIRunner(comm=comm, root=custom_root)

    assert runner_custom.get_root() == custom_root, (
        f"Rank {rank}: get_root() should return {custom_root} when provided, "
        f"but got {runner_custom.get_root()}"
    )
    assert runner_custom.is_root() == (rank == custom_root), (
        f"Rank {rank}: is_root() should return {rank == custom_root} with custom root {custom_root}, "
        f"but got {runner_custom.is_root()}"
    )


@pytest.mark.mpi(min_size=2)
def test_strict_mpi_runner_init_comm_parameter():
    """
    StrictMPIRunner.__init__() comm parameter can be given or None.
    If None, use global comm from mpi4py.
    Also verify that comm and root parameters are stored correctly.
    """
    comm_world = MPI.COMM_WORLD
    rank = comm_world.Get_rank()
    size = comm_world.Get_size()

    # Test with comm=None - should use global comm from mpi4py
    runner_none = StrictMPIRunner(comm=None)
    assert runner_none.comm is not None, (
        f"Rank {rank}: comm should not be None when using global comm"
    )
    assert runner_none.comm == comm_world, (
        f"Rank {rank}: comm should be MPI.COMM_WORLD when comm=None"
    )
    assert runner_none.root == 0, f"Rank {rank}: root should default to 0"
    assert runner_none.get_rank() == rank, (
        f"Rank {rank}: get_rank() should return correct rank"
    )
    assert runner_none.get_size() == size, (
        f"Rank {rank}: get_size() should return correct size"
    )

    # Test with comm explicitly provided
    runner_explicit = StrictMPIRunner(comm=comm_world)
    assert runner_explicit.comm == comm_world, (
        f"Rank {rank}: comm should be the provided comm"
    )
    assert runner_explicit.root == 0, f"Rank {rank}: root should default to 0"

    # Test with comm and custom root
    custom_root = 1 if size >= 2 else 0
    runner_custom_root = StrictMPIRunner(comm=comm_world, root=custom_root)
    assert runner_custom_root.comm == comm_world, (
        f"Rank {rank}: comm should be stored correctly"
    )
    assert runner_custom_root.root == custom_root, (
        f"Rank {rank}: root should be stored correctly"
    )


@pytest.mark.mpi(min_size=2)
def test_strict_mpi_runner_get_run_order():
    """
    StrictMPIRunner.get_run_order() should return the rank number (different for each rank).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    runner = StrictMPIRunner(comm=comm)

    run_order = runner.get_run_order()
    assert run_order == rank, (
        f"Rank {rank}: get_run_order() should return {rank}, but got {run_order}"
    )

    # Verify that different ranks return different run_order values
    # Gather all run_order values from all ranks
    all_run_orders = comm.allgather(run_order)
    for i, order in enumerate(all_run_orders):
        assert order == i, (
            f"Rank {i}: get_run_order() should return {i}, but got {order}"
        )
    if size > 1:
        assert len(set(all_run_orders)) == size, (
            f"All ranks should have different run_order values. Got: {all_run_orders}"
        )


@pytest.mark.mpi(min_size=2)
def test_strict_mpi_runner_call_executes_independently():
    """
    StrictMPIRunner.__call__() should execute function independently on each rank.
    Each rank should get its own result, not a broadcasted result.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    runner = StrictMPIRunner(comm=comm)

    # Test with function that takes no inputs
    def get_rank_number():
        return comm.Get_rank()

    result = runner(get_rank_number, inputs=None)
    assert result == rank, (
        f"Rank {rank}: __call__() should return {rank}, but got {result}"
    )

    # Verify that different ranks get different results
    all_results = comm.allgather(result)
    for i, res in enumerate(all_results):
        assert res == i, f"Rank {i}: __call__() should return {i}, but got {res}"
    if size > 1:
        assert len(set(all_results)) == size, (
            f"All ranks should have different results. Got: {all_results}"
        )

    # Test with function that takes inputs
    def multiply_by_rank(value):
        return value * comm.Get_rank()

    input_value = 10.0
    result_with_input = runner(multiply_by_rank, inputs=(input_value,))
    expected = input_value * rank
    assert result_with_input == expected, (
        f"Rank {rank}: __call__() with inputs should return {expected}, but got {result_with_input}"
    )

    # Verify that different ranks get different results
    all_results_with_input = comm.allgather(result_with_input)
    for i, res in enumerate(all_results_with_input):
        expected_i = input_value * i
        assert res == expected_i, (
            f"Rank {i}: __call__() with inputs should return {expected_i}, but got {res}"
        )
    if size > 1:
        assert len(set(all_results_with_input)) == size, (
            f"All ranks should have different results. Got: {all_results_with_input}"
        )
