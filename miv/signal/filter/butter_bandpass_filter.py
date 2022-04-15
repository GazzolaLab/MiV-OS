__doc__ = """
Collection of modules to filter continuous signal input.
"""
__all__ = ["ButterBandpass"]

import numpy as np
import numpy.typing as npt

import scipy.signal as sps


class ButterBandpass:
    """Classical bandpass filter using `scipy` butterworth filter."""

    def __init__(self, lowcut, highcut, order):
        """

        Parameters
        ----------
        lowcut :
            lowcut
        highcut :
            highcut
        order :
            order
        """
        pass

    def __call__(self, array: npt.ArrayLike) -> np.ndarray:
        pass

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = sps.butter(order, [low, high], btype="band")
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = sps.lfilter(b, a, data)
        return y
