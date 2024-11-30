from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mpi4py


def task_index_split(comm: mpi4py.MPI.Intercomm, num_tasks: int) -> list[int]:
    # TODO documentation
    # ex) split [1,2,3,4] --> [1,2], [3,4]

    rank = comm.Get_rank()
    size = comm.Get_size()

    tasks = np.array_split(np.arange(num_tasks), size)[rank].tolist()
    return tasks
