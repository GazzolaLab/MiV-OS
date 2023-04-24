__doc__ = """
Spikestamps
===========

.. autoclass:: Spikestamps
   :members:

"""

__all__ = ["Spikestamps"]

from typing import List, Optional

from collections.abc import Sequence

import numpy as np
import quantities as pq

from miv.core.datatype.collapsable import CollapseExtendableMixin
from miv.core.datatype.signal import Signal
from miv.core.operator.operator import DataNodeMixin


class Spikestamps(CollapseExtendableMixin, DataNodeMixin, Sequence):
    """List of array of spike times

    Represents spikes emitted by the same unit in a period of times.

    Comply with `ChannelWise` and `Extendable` protocols.
    """

    def __init__(self, iterable: Optional[List] = None):
        super().__init__()
        if iterable is None:  # Default
            iterable = []
        self.data = iterable

    @property
    def number_of_channels(self) -> int:
        """Number of channels"""
        return len(self.data)

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
        rowmax = [max(data) for data in self.data if len(data) > 0]
        return 0 if len(rowmax) == 0 else max(rowmax)

    def get_first_spikestamp(self):
        """Return timestamps of the first spike in this spikestamps"""
        rowmin = [min(data) for data in self.data if len(data) > 0]
        return 0 if len(rowmin) == 0 else min(rowmin)

    def get_view(self, t_start: float, t_end: float):
        """Truncate array and only includes spikestamps between t_start and t_end."""
        return Spikestamps(
            [
                np.array(sorted(list(filter(lambda x: t_start <= x <= t_end, arr))))
                for arr in self.data
            ]
        )

    def select(self, indices, keepdims: bool = True):
        """Select channels by indices."""
        if keepdims:
            data = [
                self.data[idx] if idx in indices else []
                for idx in range(self.number_of_channels)
            ]
            return Spikestamps(data)
        else:
            return Spikestamps([self.data[idx] for idx in indices])

    def neo(self):
        """Cast to neo.SpikeTrain"""
        import neo

        t_start = self.get_first_spikestamp()
        t_stop = self.get_last_spikestamp()
        return [
            neo.SpikeTrain(arr, t_start=t_start, t_stop=t_stop, units=pq.s)
            for arr in self.data
        ]

    def binning(
        self,
        bin_size: float = 1 * pq.ms,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        return_count: bool = False,
    ) -> Signal:
        """
        Forms a binned spiketrain using the spiketrain

        Parameters
        ----------
        bin_size : float | pq.Quantity
            bin size in the unit of time.
        return_count : bool
            If set to true, return the bin count. (default=False)

        Returns
        -------
        bin_spike: numpy.ndarray
            binned spiketrain with 1 corresponding to spike and zero otherwise

        """
        spiketrain = self.data

        if isinstance(bin_size, pq.Quantity):
            bin_size = bin_size.rescale(pq.s).magnitude
        assert bin_size > 0, "bin size should be greater than 0"

        t_start = self.get_first_spikestamp() if t_start is None else t_start
        t_end = self.get_last_spikestamp() if t_end is None else t_end
        assert (
            t_start < t_end
        ), f"t_start ({t_start}) should be less than t_end ({t_end})"
        n_bins = int(np.ceil((t_end - t_start) / bin_size))
        time = t_start + (np.arange(n_bins + 1) * bin_size)

        num_channels = self.number_of_channels
        signal = Signal(
            data=np.zeros(
                [time.shape[0] - 1, num_channels],
                dtype=np.int_ if return_count else np.bool_,
            ),
            timestamps=time[:-1],
            rate=1.0 / bin_size,
        )
        for idx, spiketrain in enumerate(self.data):
            bins = np.digitize(spiketrain, time)
            bincount = np.bincount(bins, minlength=n_bins + 2)[1:-1]
            if return_count:
                bin_spike = bincount
            else:
                bin_spike = (bincount != 0).astype(np.bool_)
            signal.data[:, idx] = bin_spike
        return signal
