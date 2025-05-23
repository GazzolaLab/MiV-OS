__doc__ = ""
__all__ = ["ButterBandpass"]

import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sps

from miv.core.datatype import Signal
from miv.core.operator_generator.operator import GeneratorOperatorMixin
from miv.core.operator_generator.wrapper import cache_generator_call


@dataclass
class ButterBandpass(GeneratorOperatorMixin):
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

    lowcut: float | None = None
    highcut: float | None = None
    order: int = 5
    tag: str = "bandpass filter"
    btype: str = "bandpass"

    @cache_generator_call
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

        durations = []
        for ch in range(num_channel):
            stime = time.time()
            y[:, ch] = sps.sosfiltfilt(sos, signal.data[:, ch])

            # logging
            duration = time.time() - stime
            durations.append(duration)
            # self.logger.info(f"Filtering channel {ch} took {duration:.3f} sec.")
        # log statistics - average, min, max duration
        self.logger.info(
            f"Filtering took {np.mean(durations):.3f} sec on average. "
            f"Min: {np.min(durations):.3f} sec, "
            f"Max: {np.max(durations):.3f} sec."
        )
        return Signal(data=y, timestamps=signal.timestamps, rate=rate)

    def __post_init__(self):
        if self.lowcut is not None and self.highcut is not None:
            assert self.lowcut < self.highcut, (
                f"{self.lowcut=} should be lower than {self.highcut=}."
            )
            assert min(self.lowcut, self.highcut) > 0, (
                "Filter critical frequencies must be greater than 0"
            )
        elif self.lowcut is not None:
            assert self.lowcut > 0, "Filter frequencies must be greater than 0"
        elif self.highcut is not None:
            assert self.highcut > 0, "Filter frequencies must be greater than 0"
        assert self.order > 0 and isinstance(self.order, int), (
            f"Filter {self.order} must be an nonnegative integer."
        )
        assert self.lowcut is not None or self.highcut is not None, (
            "Filtering frequency cannot be both None"
        )
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

    def firstiter_plot_frequency_response(
        self,
        signal,
        inputs,
        show=False,
        save_path=None,
    ):
        """plot_frequency_response"""
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
        plt.ylim(-50, 10)

        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "filter_frequency_response.png"))

        plt.close(fig)
