__all__ = ["Spikestamps"]

from typing import Optional

from collections import UserList

import numpy as np


class Spikestamps(UserList):
    """List of array of spike times

    Represents spikes emitted by the same unit in a period of times.
    """

    def __init__(self, iterable):
        super().__init__()
        self.data = iterable

    def __setitem__(self, index, item):
        self.data[index] = item

    def insert(self, index, item):
        if index > len(self.data) or index < 0:
            raise IndexError("Index out of range")
        self.data.insert(index, item)

    def append(self, item):
        self.data.append(item)

    def extend(self, other):
        if isinstance(other, type(self)):
            length_diff = len(other) - len(self.data)
            if length_diff > 0:
                for _ in range(length_diff):
                    self.data.append([])
            for idx, array in enumerate(other):
                self.data[idx].extend(array)
        else:
            self.data.extend(item for item in other)
