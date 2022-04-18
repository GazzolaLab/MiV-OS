__all__ = ["SpikeDetectionProtocol"]

from typing import Protocol

import neo.core

from miv.typing import SignalType, TimestampsType, SpikestampsType


class SpikeDetectionProtocol(Protocol):
    def __call__(
        self, signal: SignalType, timestamps: TimestampsType, sampling_rate: float
    ) -> SpikestampsType:
        ...

    def __repr__(self) -> str:
        ...
