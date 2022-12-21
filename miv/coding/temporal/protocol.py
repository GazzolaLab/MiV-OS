__all__ = ["TemporalEncoderProtocol", "SpikerAlgorithmProtocol"]

from typing import Optional, Protocol, Tuple

import numpy as np

from miv.typing import SignalType, SpikestampsType, TimestampsType


class TemporalEncoderProtocol(Protocol):
    def __call__(self, signal: np.ndarray) -> SignalType:  # pragma: no cover
        ...


class SpikerAlgorithmProtocol(Protocol):
    def __init__(self):  # pragma: no cover
        ...

    def __call__(
        self, signal: np.ndarray, sample_rate
    ) -> Tuple[SpikestampsType, TimestampsType]:  # pragma: no cover
        ...

    def save(self) -> None:  # pragma: no cover
        ...
