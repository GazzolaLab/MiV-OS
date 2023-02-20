__doc__ = ""
__all__ = ["FilterProtocol"]

import typing
from typing import Callable, Optional, Protocol, Union

import numpy as np
import numpy.typing as npt

from miv.core.datatype import Signal
from miv.typing import SignalType


class FilterProtocol(Protocol):
    """Behavior definition of all filter operator."""

    tag: str = ""

    def __call__(self, array: Signal) -> Signal:
        """User can apply the filter by directly calling.
        Parameters
        ----------
        array : Signal
        """
        ...

    def __repr__(self) -> str:
        """String representation for interactive debugging."""
        ...
