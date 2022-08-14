__all__ = ["TemporalEncoderProtocol", "SpikerAlgorithmProtocol"]

from typing import Optional, Protocol, Tuple

import numpy as np

from miv.typing import SignalType, SpikestampsType, TimestampsType


class TemporalEncoderProtocol(Protocol):
    def __call__(self, signal: np.ndarray) -> SignalType:
        ...


class SpikerAlgorithmProtocol(Protocol):
    def __init__(self):
        ...

    def __call__(
        self, signal: np.ndarray, sample_rate
    ) -> Tuple[SpikestampsType, TimestampsType]:
        ...

    def save(self) -> None:
        ...
