__doc__ = """
Signal
======

.. autoclass:: Signal
   :members:

"""

__all__ = ["Signal"]

from typing import Optional, Tuple

from dataclasses import dataclass

import numpy as np

from miv.core.operator.operator import DataNodeMixin
from miv.core.policy import SupportMultiprocessing
from miv.typing import SignalType, TimestampsType


@dataclass
class Signal(SupportMultiprocessing, DataNodeMixin):
    """
    Contiguous array of raw signal type.

    [signal length, number of channels]
    """

    _CHANNELAXIS = 1
    _SIGNALAXIS = 0

    data: SignalType
    timestamps: TimestampsType
    rate: int = 30_000

    def __post_init__(self):
        super().__init__()
        self.data = np.asarray(self.data)
        assert len(self.data.shape) == 2, "Signal must be 2D array"

    @property
    def number_of_channels(self) -> int:
        """Number of channels in the signal."""
        return self.data.shape[self._CHANNELAXIS]

    def __getitem__(self, i: int) -> SignalType:
        return self.data[:, i]  # TODO: Fix to row-major

    def select(self, indices: Tuple[int, ...]) -> "Signal":
        """Select channels by indices."""
        return Signal(self.data[:, indices], self.timestamps, self.rate)

    def get_start_time(self):
        """Get the start time of the signal."""
        return self.timestamps.min()

    def get_end_time(self):
        """Get the end time of the signal."""
        return self.timestamps.max()

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the signal."""
        return self.data.shape

    def append(self, value) -> None:
        """Append a channels to the end of the existing signal."""
        assert value.shape[self._SIGNALAXIS] == self.data.shape[self._SIGNALAXIS]
        self.data = np.append(self.data, value, axis=self._CHANNELAXIS)

    def extend_signal(self, value: SignalType, time: TimestampsType) -> None:
        """Append a signal to the end of the existing signal."""
        assert value.shape[self._SIGNALAXIS] == time.shape[0]
        assert (
            value.shape[self._CHANNELAXIS] == self.data.shape[self._CHANNELAXIS]
        ), "Signal must have same number of channels"
        self.data = np.append(self.data, value, axis=self._SIGNALAXIS)
        self.timestamps = np.append(self.timestamps, time)

    def prepend_signal(self, value: SignalType, time: TimestampsType) -> None:
        """Prepend a signal to the end of the existing signal."""
        assert (
            value.shape[self._SIGNALAXIS] == time.shape[0]
        ), "Time and signal must have same length"
        assert (
            value.shape[self._CHANNELAXIS] == self.data.shape[self._CHANNELAXIS]
        ), "Signal must have same number of channels"
        self.data = np.append(value, self.data, axis=self._SIGNALAXIS)
        self.timestamps = np.append(self.timestamps, time)
