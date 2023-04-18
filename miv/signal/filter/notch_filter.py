__doc__ = ""
__all__ = ["Notch"]

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
class Notch(OperatorMixin):
    """Notch filter using `scipy` second-order notch filter, wrapped in operator mixin.

    Parameters
    ----------
    w0 : float
        notch frequency
    Q : float
        Q factor
    tag : str
        Tag for the collection of filter.
    """

    f0: Optional[float] = 60.0  # Hz
    Q: Optional[float] = 30.0
    tag: str = "notch filter"

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
        b, a = sps.iirnotch(w0=self.f0, Q=self.Q, fs=rate)
        y = signal.data.copy()
        num_channel = signal.number_of_channels
        for ch in range(num_channel):
            y[:, ch], h = sps.freqz(b, a, signal.data[:, ch])
        return Signal(data=y, timestamps=signal.timestamps, rate=rate)

    def __post_init__(self):
        super().__init__()
        self.cacher.policy = "OFF"

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
