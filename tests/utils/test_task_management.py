import pytest

from miv.utils.mpi.task_management import task_index_split


class FakeComm:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._size


@pytest.mark.parametrize(
    "num_tasks,size,expected",
    [
        (4, 2, [[0, 1], [2, 3]]),
        (5, 3, [[0, 1], [2, 3], [4]]),
    ],
)
def test_task_index_split_even_and_uneven(num_tasks, size, expected):
    for rank in range(size):
        comm = FakeComm(rank, size)
        assert task_index_split(comm, num_tasks) == expected[rank]


def test_task_index_split_rejects_negative_tasks():
    comm = FakeComm(rank=0, size=2)
    with pytest.raises(ValueError, match="num_tasks must be non-negative"):
        task_index_split(comm, -1)
