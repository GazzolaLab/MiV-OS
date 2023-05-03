__doc__ = ""
__all__ = ["ButterBandpass"]

from typing import Optional

import inspect
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
        sos = self._butter_bandpass(rate)
        y = signal.data.copy()
        num_channel = signal.number_of_channels
        for ch in range(num_channel):
            y[:, ch] = sps.sosfiltfilt(sos, signal.data[:, ch])
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
        self.cacher.policy = "OFF"

    def _butter_bandpass(self, sampling_rate: float):
        if self.btype in ["bandpass", "bandstop"]:
            low = self.lowcut
            high = self.highcut
            critical_frequency = [low, high]
        elif self.btype == "highpass":
            critical_frequency = self.lowcut
        elif self.btype == "lowpass":
            critical_frequency = self.highcut
        else:
            raise ValueError("Unknown btype: %s" % self.btype)
        sos = sps.butter(
            self.order,
            critical_frequency,
            fs=sampling_rate,
            btype=self.btype,
            output="sos",
        )
        return sos

    def _generator_plot_frequency_response(
        self, signal, show=False, save_path=None, index=None, zipped_inputs=None
    ):
        """plot_frequency_response"""
        if index > 0:
            return
        rate = signal.rate
        sos = self._butter_bandpass(rate)
        w, h = sps.sosfreqz(sos, worN=2000, fs=rate)
        ah = np.abs(h)

        fig = plt.figure()
        plt.semilogx(w, 20 * np.log10(np.where(ah > 0.0, ah, 1e-300)))
        plt.title("Butterworth filter frequency response")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude [dB]")
        plt.margins(0, 0.1)
        plt.grid(which="both", axis="both")
        if self.lowcut is not None:
            plt.axvline(self.lowcut, color="red")
        if self.highcut is not None:
            plt.axvline(self.highcut, color="red")

        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "filter_frequency_response.png"))

        plt.close(fig)
