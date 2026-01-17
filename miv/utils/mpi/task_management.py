from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mpi4py


def task_index_split(comm: mpi4py.MPI.Comm, num_tasks: int) -> list[int]:
    """Split task indices evenly across MPI ranks.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Active MPI communicator used to query rank and size.
    num_tasks : int
        Total number of tasks to split. Must be non-negative.

    Returns
    -------
    list[int]
        Task indices assigned to the current rank (0-indexed).

    Notes
    -----
    Tasks are divided as evenly as possible, with lower ranks receiving at most
    one additional task when ``num_tasks`` is not divisible by ``size``. For
    example, with ``num_tasks=5`` and ``size=3``, ranks receive ``[0, 1]``,
    ``[2, 3]``, and ``[4]`` respectively.
    """

    if num_tasks < 0:
        raise ValueError("num_tasks must be non-negative")

    rank = comm.Get_rank()
    size = comm.Get_size()

    base, remainder = divmod(num_tasks, size)
    start = rank * base + min(rank, remainder)
    stop = start + base + (1 if rank < remainder else 0)
    return list(range(start, stop))
