__doc__ = """
Filter collection.
"""
__all__ = ["FilterCollection"]

from typing import Union
import numpy as np
import numpy.typing as npt

from collections.abc import Callable, MutableSequence

from miv.typing import SignalType


class FilterCollection(MutableSequence, Callable):
    """Create sequence of multiple filters.

    Each appended filters are expected to abide the FilterProtocol.

        Parameters
        ----------
        tag : str
            Tag for the collection of filter.

        Examples
        --------


    """

    def __init__(self, tag: str = ""):
        self.filters = []
        self.tag = tag

    # Callable abstract methods
    def __call__(self, signal: SignalType) -> SignalType:
        for filter in self.filters:
            signal = filter(signal)
        return signal

    # MutableSequence abstract methods
    def __len__(self):
        return len(self.filters)

    def __getitem__(self, idx):
        return self.filters[idx]

    def __delitem__(self, idx):
        del self.filters[idx]

    def __setitem__(self, idx, system):
        self.filters[idx] = system

    def insert(self, idx, system):
        self.filters.insert(idx, system)

    def __repr__(self):
        s = f"Collection of filters({self.tag=}, {id(self)=})\n"
        for filter in self.filters:
            s += "  " + filter.__repr__() + "\n"
        return s
