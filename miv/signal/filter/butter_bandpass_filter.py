__doc__ = ""
__all__ = ["ButterBandpass"]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import scipy.signal as sps

from miv.typing import SignalType


@dataclass(frozen=True)
class ButterBandpass:
    """Classical bandpass filter using `scipy` butterworth filter

    Parameters
    ----------
    lowcut : float
        low-pass frequency
    highcut : float
        high-pass frequency
    order : int
        The order of the filter. (default=5)
    tag : str
        Tag for the collection of filter.
    """

    lowcut: float
    highcut: float
    order: int = 5
    tag: str = ""

    def __call__(self, signal: SignalType, sampling_rate: float) -> SignalType:
        b, a = self._butter_bandpass(sampling_rate)
        y = sps.lfilter(b, a, signal)
        return y

    def _butter_bandpass(self, sampling_rate: float):
        nyq = 0.5 * sampling_rate
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = sps.butter(self.order, [low, high], btype="band")
        return b, a
