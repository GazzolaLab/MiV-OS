import numpy as np
import pytest

from miv.core.datatype import CollapseExtendableMixin


class CollapsableData(CollapseExtendableMixin):
    def __init__(self):
        self.data = []

    def extend(self, values):
        self.data.extend(values)

    def append(self, item):
        self.data.append(item)

    def __setitem__(self, index, item):
        self.data[index] = item

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def insert(self, index, item):
        if index > len(self.data) or index < 0:
            raise IndexError("Index out of range")
        self.data.insert(index, item)


def test_support_mpi_merge_policy():
    from mpi4py import MPI

    from miv.core.policy import SupportMPIMerge

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    scale = 5  # arbitrary

    def func():
        data = CollapsableData()
        data.data = [rank * scale]
        return data

    policy = SupportMPIMerge()
    result = policy(func)
    print(f"rank: {rank}, result: {result.data}")

    assert result.data == (np.arange(size) * scale).tolist()
