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

    def plot_frequency_response(self, signal, show=False, save_path=None):
        """plot_frequency_response"""
        rate = next(signal).rate
        b, a = sps.iirnotch(w0=self.f0, Q=self.Q, fs=rate)
        freq, h = sps.freqz(b, a, fs=rate)

        fig, ax = plt.subplots(2, 1, figsize=(8, 6))

        ax[0].plot(freq, 20 * np.log10(abs(h)))
        ax[0].set_title("Frequency Response")
        ax[0].set_ylabel("Amplitude (dB)")
        ax[0].set_xlim([0, 100])
        ax[0].set_ylim([-25, 10])
        ax[0].grid(True)

        ax[1].plot(freq, np.unwrap(np.angle(h)) * 180 / np.pi)
        ax[1].set_ylabel("Angle (degrees)")
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_xlim([0, 100])
        ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
        ax[1].set_ylim([-90, 90])
        ax[1].grid(True)

        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "filter_frequency_response.png"))

        plt.close(fig)
