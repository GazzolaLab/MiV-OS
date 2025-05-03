__doc__ = """
Signal
======

.. autoclass:: Signal
   :members:

"""

__all__ = ["Signal"]

from typing import cast

import pickle
from dataclasses import dataclass

import numpy as np

from miv.core.datatype.mixin_colapsable import ConcatenateMixin
from miv.core.operator.operator import DataNodeMixin
from miv.core.operator.policy import SupportMultiprocessing
from miv.typing import SignalType, SpikestampsType


@dataclass
class Signal(SupportMultiprocessing, DataNodeMixin, ConcatenateMixin):
    """
    Contiguous array of raw signal type.

    [signal length, number of channels]
    """

    # This choice of axis is mainly due to the memory storage structure.
    _CHANNELAXIS = 1
    _SIGNALAXIS = 0

    data: SignalType
    timestamps: SpikestampsType
    rate: float = 30_000

    def __post_init__(self) -> None:
        super().__init__()
        self.data = np.asarray(self.data)
        assert len(self.data.shape) == 2, "Signal must be 2D array"

    @property
    def number_of_channels(self) -> int:
        """Number of channels in the signal."""
        return int(self.data.shape[self._CHANNELAXIS])

    def __getitem__(self, i: int) -> SignalType:
        return self.data[:, i]

    def select(self, indices: tuple[int, ...]) -> "Signal":
        """Select channels by indices."""
        return Signal(self.data[:, indices], self.timestamps, self.rate)

    def get_start_time(self) -> float:
        """Get the start time of the signal."""
        return cast(float, self.timestamps.min())

    def get_end_time(self) -> float:
        """Get the end time of the signal."""
        return cast(float, self.timestamps.max())

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the signal."""
        return cast(tuple[int, int], self.data.shape)

    def append(self, value: np.ndarray) -> None:
        """Append a channels to the end of the existing signal."""
        assert value.shape[self._SIGNALAXIS] == self.data.shape[self._SIGNALAXIS]
        self.data = np.append(self.data, value, axis=self._CHANNELAXIS)

    def extend_signal(self, data: np.ndarray, time: SpikestampsType) -> None:
        """Append a signal to the end of the existing signal."""
        assert data.shape[self._SIGNALAXIS] == time.shape[0]
        assert data.shape[self._CHANNELAXIS] == self.data.shape[self._CHANNELAXIS], (
            "Signal must have same number of channels"
        )
        self.data = np.append(self.data, data, axis=self._SIGNALAXIS)
        self.timestamps = np.append(self.timestamps, time)

    def extend(self, value: "Signal") -> None:
        """Append a signal to the end of the existing signal."""
        self.extend_signal(value.data, value.timestamps)

    def prepend_signal(self, data: np.ndarray, time: SpikestampsType) -> None:
        """Prepend a signal to the end of the existing signal."""
        assert data.shape[self._SIGNALAXIS] == time.shape[0], (
            "Time and signal must have same length"
        )
        assert data.shape[self._CHANNELAXIS] == self.data.shape[self._CHANNELAXIS], (
            "Signal must have same number of channels"
        )
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
            return cast(Signal, pickle.load(f))
