__doc__ = ""
__all__ = ["ButterBandpass"]

from dataclasses import dataclass

import matplotlib.pyplot as plt
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

    def __call__(
        self,
        signal: SignalType,
        sampling_rate: float,
        **kwargs,
    ) -> SignalType:
        """__call__.

        Parameters
        ----------
        signal : SignalType
            signal
        sampling_rate : float
            sampling_rate

        Returns
        -------
        SignalType

        """
        b, a = self._butter_bandpass(sampling_rate)
        y = signal.copy()
        if len(signal.shape) == 1:
            y = sps.lfilter(b, a, signal)
        elif len(signal.shape) == 2:
            for ch in range(signal.shape[1]):
                y[:, ch] = sps.lfilter(b, a, signal[:, ch])
        else:
            raise ValueError(
                "This filter can be only applied to 1D (signal) or 2D array (signal, channel)"
            )
        return y

    def __post_init__(self):
        assert (
            self.lowcut < self.highcut
        ), f"{self.lowcut=} should be lower than {self.highcut=}."
        assert self.order > 0 and isinstance(
            self.order, int
        ), f"Filter {self.order} must be an nonnegative integer."
        assert (
            min(self.lowcut, self.highcut) > 0
        ), "Filter critical frequencies must be greater than 0"

    def _butter_bandpass(self, sampling_rate: float):
        nyq = 0.5 * sampling_rate
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = sps.butter(self.order, [low, high], btype="band")
        return b, a

    def plot_frequency_response(self, sampling_rate: float):
        """plot_frequency_response

        Parameters
        ----------
        sampling_rate : float

        Returns
        -------
        plt.Figure
        """
        b, a = self._butter_bandpass(sampling_rate)
        w, h = sps.freqs(b, a)
        fig = plt.figure()
        plt.semilogx(w, 20 * np.log10(abs(h)))
        plt.title(
            f"Butterworth filter (order{self.order}) frequency response [{self.lowcut},{self.highcut}]"
        )
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        return fig
