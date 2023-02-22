__doc__ = ""
__all__ = ["ButterBandpass"]

from typing import Optional

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.signal as sps

from miv.core.datatype import Signal
from miv.core.operator import OperatorMixin
from miv.core.wrapper import wrap_generator_to_generator
from miv.typing import SignalType


@dataclass
class ButterBandpass(OperatorMixin):
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
    btype : str
        Filter type: bandpass, lowpass, highpass, bandstop. (default="bandpass")
        If set to lowpass, the critical frequency is set to 'highcut'.
        If set to highpass, the critical frequency is set to 'lowcut'.
        If set to bandpass or bandstop, the critical frequency is '[lowcut, highcut]'.
    """

    lowcut: Optional[float] = None
    highcut: Optional[float] = None
    order: int = 5
    tag: str = "bandpass filter"
    btype: str = "bandpass"

    @wrap_generator_to_generator
    def __call__(self, signal: Signal) -> Signal:
        """__call__.

        Parameters
        ----------
        signal : SignalType
            signal

        Returns
        -------
        Signal

        """
        rate = signal.rate
        b, a = self._butter_bandpass(rate)
        y = signal.data.copy()
        num_channel = signal.number_of_channels
        for ch in range(num_channel):
            y[:, ch] = sps.lfilter(b, a, signal.data[:, ch])
        return Signal(data=y, timestamps=signal.timestamps, rate=rate)

    def __post_init__(self):
        if self.lowcut is not None and self.highcut is not None:
            assert (
                self.lowcut < self.highcut
            ), f"{self.lowcut=} should be lower than {self.highcut=}."
            assert (
                min(self.lowcut, self.highcut) > 0
            ), "Filter critical frequencies must be greater than 0"
        elif self.lowcut is not None:
            assert self.lowcut > 0, "Filter frequencies must be greater than 0"
        elif self.highcut is not None:
            assert self.highcut > 0, "Filter frequencies must be greater than 0"
        assert self.order > 0 and isinstance(
            self.order, int
        ), f"Filter {self.order} must be an nonnegative integer."
        assert (
            self.lowcut is not None or self.highcut is not None
        ), "Filtering frequency cannot be both None"
        super().__init__()

    def _butter_bandpass(self, sampling_rate: float):
        nyq = 0.5 * sampling_rate
        if self.btype in ["bandpass", "bandstop"]:
            low = self.lowcut / nyq
            high = self.highcut / nyq
            critical_frequency = [low, high]
        elif self.btype == "highpass":
            critical_frequency = self.lowcut / nyq
        elif self.btype == "lowpass":
            critical_frequency = self.highcut / nyq
        else:
            raise ValueError("Unknown btype: %s" % self.btype)
        b, a = sps.butter(self.order, critical_frequency, btype=self.btype)
        return b, a

    # def plot_frequency_response(self, signal:Signal, show=False, save_path=None):
    #     """plot_frequency_response

    #     Parameters
    #     ----------
    #     signal : Signal

    #     Returns
    #     -------
    #     plt.Figure
    #     """
    #     sampling_rate = next(signal).rate
    #     b, a = self._butter_bandpass(sampling_rate)
    #     w, h = sps.freqs(b, a)
    #     fig = plt.figure()
    #     plt.semilogx(w, 20 * np.log10(abs(h)))
    #     plt.title(
    #         f"Butterworth filter (order{self.order}) frequency response [{self.lowcut},{self.highcut}]"
    #     )
    #     plt.xlabel("Frequency")
    #     plt.ylabel("Amplitude")
    #     if show:
    #         plt.show()
    #     if save_path is not None:
    #         fig.savefig(os.path.join(save_path, 'filter_frequency_response.png'))
    #     return fig
