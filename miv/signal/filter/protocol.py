__doc__ = ""
__all__ = ["FilterProtocol"]

import typing
from typing import Union, Callable, Optional
from typing import Protocol

import numpy as np
import numpy.typing as npt

from miv.typing import SignalType


class FilterProtocol(Protocol):
    """Behavior definition of all filter operator."""

    def __call__(self, array: SignalType, sampling_rate: float, **kwargs) -> SignalType:
        """User can apply the filter by directly calling.
        Parameters
        ----------
        array : SignalType {"numpy.ndarray", "neo.core.AnalogSignal"}
        samping_rate : float
        """
        ...

    @property
    def tag(self) -> str:
        ...
