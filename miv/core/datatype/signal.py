__doc__ = """
Signal
======

.. autoclass:: Signal
   :members:

"""

__all__ = ["Signal"]

from typing import Optional, Tuple

import pickle
from dataclasses import dataclass

import numpy as np

from miv.core.datatype.collapsable import CollapseExtendableMixin
from miv.core.operator.operator import DataNodeMixin
from miv.core.operator.policy import SupportMultiprocessing
from miv.typing import SignalType, TimestampsType


@dataclass
class Signal(SupportMultiprocessing, DataNodeMixin, CollapseExtendableMixin):
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

    def extend_signal(self, data: np.ndarray, time: TimestampsType) -> None:
        """Append a signal to the end of the existing signal."""
        assert data.shape[self._SIGNALAXIS] == time.shape[0]
        assert (
            data.shape[self._CHANNELAXIS] == self.data.shape[self._CHANNELAXIS]
        ), "Signal must have same number of channels"
        self.data = np.append(self.data, data, axis=self._SIGNALAXIS)
        self.timestamps = np.append(self.timestamps, time)

    def prepend_signal(self, data: np.ndarray, time: TimestampsType) -> None:
        """Prepend a signal to the end of the existing signal."""
        assert (
            data.shape[self._SIGNALAXIS] == time.shape[0]
        ), "Time and signal must have same length"
        assert (
            data.shape[self._CHANNELAXIS] == self.data.shape[self._CHANNELAXIS]
        ), "Signal must have same number of channels"
        self.data = np.append(data, self.data, axis=self._SIGNALAXIS)
        self.timestamps = np.append(self.timestamps, time)

    def save(self, path: str) -> None:
        """Save signal to file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Signal":
        """Load signal from file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def from_collapse(cls, values):
        obj = None
        for idx, value in enumerate(values):
            if idx == 0:
                obj = value
            else:
                obj.extend_signal(value.data, value.timestamps)
        return obj
