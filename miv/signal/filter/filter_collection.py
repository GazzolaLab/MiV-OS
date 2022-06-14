__doc__ = """

Signal Filter
#############

.. currentmodule:: miv.signal.filter

A sequence of filter can be composed using the module `FilterCollection`.

.. autoclass:: FilterCollection
   :members: append, insert

.. autosummary::
   :nosignatures:
   :toctree: _toctree/FilterAPI

   FilterProtocol
   ButterBandpass
   MedianFilter

"""
__all__ = ["FilterCollection"]

from typing import TYPE_CHECKING, List

from collections.abc import MutableSequence

import numpy as np
import numpy.typing as npt

from miv.typing import SignalType

if TYPE_CHECKING:
    from miv.signal.filter import FilterProtocol


class FilterCollection(MutableSequence):
    """Create sequence of multiple filters.

    Each appended filters are expected to abide the FilterProtocol.

        Parameters
        ----------
        tag : str
            Tag for the collection of filter.

        See Also
        --------
        miv.signal.filter.ButterBandpass : butterworth filter
        miv.signal.filter.MedianFilter : median filter with threshold

        Examples
        --------
        >>> custom_filter = (
        ...     FilterCollection(tag="custom filter 1")
        ...       .append(ButterBandpass(lowcut=300, highcut=3000, order=3))
        ...       .append(MedianFilter(threshold=50, k=50))
        ... )
        >>> custom_filter
        Collection of filters(self.tag='custom filter 1', id(self)=140410408570736)
          0: ButterBandpass(lowcut=300, highcut=3000, order=3, tag='')
          1: MedianFilter(threshold=50, k=50, tag='')
        >>> filtered_signal = custom_filter(signal=signal, sampling_rate=30_000)

    """

    def __init__(self, tag: str = ""):
        self.filters: List[FilterProtocol] = []
        self.tag: str = tag

    def __call__(self, signal: SignalType, sampling_rate: float) -> SignalType:
        for filt in self.filters:
            signal = filt(signal, sampling_rate)
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
        """

        Parameters
        ----------
        idx : int
        system : FilterProtocol

        Returns
        -------
        self : FilterCollection
            User can chain multiple insert/append to build sequence of filter.
            Check the example above.

        """

        self.filters.insert(idx, system)
        return self

    def append(self, system):
        """

        Parameters
        ----------
        system : FilterProtocol

        Returns
        -------
        self : FilterCollection
            User can chain multiple insert/append to build sequence of filter.
            Check the example above.

        """
        self.filters.append(system)
        return self

    def __repr__(self):
        s = f"Collection of filters({self.tag=}, {id(self)=})\n"
        for idx, filter in enumerate(self.filters):
            s += f"  {idx}: {filter}\n"
        return s
