__doc__ = """
Spikestamps
===========

.. autoclass:: Spikestamps
   :members:

"""

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
        """
        Extend spikestamps from another `Spikestamps` or list of arrays.

        If the given parameter is another `Spikestamps`, each arrays are concatenated.
        Remaining channels will be added as different channels.

        If the given parameter is list of another arrays, each arrays will be added as different channels.

        Examples
        --------

        >>> a = Spikestamps([[0,1,2],[0,3]])
        >>> b = Spikestamps([[3,5],[4],[0,1]])
        >>> a.extend(b)
        >>> a
        [[0,1,2,3,5], [0,3,4], [0,1]]

        >>> c = [[1],[5],[0]]
        >>> a.extend(c)
        >>> a
        [[0,1,2,3,5], [0,3,4], [0,1], [1], [5], [0]]

        """
        if isinstance(other, type(self)):
            length_diff = len(other) - len(self.data)
            if length_diff > 0:
                for _ in range(length_diff):
                    self.data.append([])
            for idx, array in enumerate(other):
                self.data[idx].extend(array)
        else:
            self.data.extend(item for item in other)

    def get_count(self):
        """Return list of spike-counts for each channel."""
        return [len(data) for data in self.data]

    def get_last_spikestamp(self):
        """Return timestamps of the last spike in this spikestamps"""
        return max([data[-1] for data in self.data if len(data) > 0])

    def get_first_spikestamp(self):
        """Return timestamps of the first spike in this spikestamps"""
        return min([data[0] for data in self.data if len(data) > 0])

    def get_view(self, tstart: float, tend: float):
        """Truncate array and only includes spikestamps between tstart and tend."""
        return Spikestamps(
            [
                np.array(sorted(list(filter(lambda x: tstart <= x <= tend, arr))))
                for arr in self.data
            ]
        )
