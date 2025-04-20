__doc__ = """
Events
======

.. autoclass:: Events
   :members:

"""

__all__ = ["Events"]

from typing import cast


import numpy as np
import quantities as pq

from miv.core.datatype.mixin_colapsable import ConcatenateMixin
from miv.core.datatype.signal import Signal
from miv.core.operator.operator import DataNodeMixin


class Events(ConcatenateMixin, DataNodeMixin):
    """
    A list of events.

    Comply with `Extendable` protocols.
    """

    def __init__(self, data: list[float] | None = None) -> None:
        super().__init__()
        self.data = np.asarray(data) if data is not None else np.array([])

    def append(self, item: float) -> None:
        self.data = np.append(self.data, item)

    def extend(self, other: "Events") -> None:
        self.data = np.append(self.data, other.data)

    def __len__(self) -> int:
        return len(self.data)

    def get_last_event(self) -> float:
        """Return timestamps of the last event"""
        return cast(float, max(self.data))

    def get_first_event(self) -> float:
        """Return timestamps of the first event"""
        return cast(float, min(self.data))

    def get_view(self, t_start: float, t_end: float) -> "Events":
        """Truncate array and only includes spikestamps between t_start and t_end."""
        return Events(sorted(filter(lambda x: t_start <= x <= t_end, self.data)))

    def binning(
        self,
        bin_size: float | pq.Quantity = 0.001,
        t_start: float | None = None,
        t_end: float | None = None,
        return_count: bool = False,
    ) -> Signal:
        """
        Forms a binned events

        Parameters
        ----------
        bin_size : float | pq.Quantity
            bin size in the unit of time.
        return_count : bool
            If set to true, return the bin count. (default=False)
        """

        if isinstance(bin_size, pq.Quantity):
            bin_size = bin_size.rescale(pq.s).magnitude
        assert bin_size > 0, "bin size should be greater than 0"

        t_start = self.get_first_event() if t_start is None else t_start
        t_end = self.get_last_event() if t_end is None else t_end
        n_bins = int(np.ceil((t_end - t_start) / bin_size))
        time = t_start + (np.arange(n_bins + 1) * bin_size)

        signal = Signal(
            data=np.zeros(
                [time.shape[0] - 1, 1],
                dtype=np.int_ if return_count else np.bool_,
            ),
            timestamps=time[:-1],
            rate=1.0 / bin_size,
        )

        # TODO: Make separate free function for this binning process
        bins = np.digitize(self.data, time)
        bincount = np.bincount(bins, minlength=n_bins + 2)[1:-1]
        if return_count:
            bin_events = bincount
        else:
            bin_events = (bincount != 0).astype(np.bool_)
        signal.data[:, 0] = bin_events

        return signal
