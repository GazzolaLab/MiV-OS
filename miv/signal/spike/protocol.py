__all__ = [
    "SpikeDetectionProtocol",
    "SpikeFeatureExtractionProtocol",
    "UnsupervisedFeatureClusteringProtocol",
]

from typing import Protocol, Union, Any, Iterable

import numpy as np
import neo.core

from miv.typing import SignalType, TimestampsType, SpikestampsType


class SpikeDetectionProtocol(Protocol):
    def __call__(
        self, signal: SignalType, timestamps: TimestampsType, sampling_rate: float
    ) -> SpikestampsType:
        ...

    def __repr__(self) -> str:
        ...


# TODO: Behavior is clear, but not sure what the name should be
# class SpikeSortingProtocol(Protocol):
#    def __call__(
#        self, signals: Iterable[SignalType]
#    ) -> np.ndarray:
#        ...
#
#    def __repr__(self) -> str:
#        ...


class SpikeFeatureExtractionProtocol(Protocol):
    """ex) wavelet transform, PCA, etc."""

    def __repr__(self) -> str:
        ...


class UnsupervisedFeatureClusteringProtocol(Protocol):
    def __repr__(self) -> str:
        ...

    def fit(self, X: np.ndarray):
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...
