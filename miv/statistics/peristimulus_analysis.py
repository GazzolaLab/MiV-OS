__all__ = ["PSTH", "PeristimulusActivity", "peri_stimulus_time", "PSTHOverlay"]

from typing import List

import itertools
import os
import sys
from dataclasses import dataclass
from glob import glob

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from tqdm import tqdm

from miv.core.datatype import Signal, Spikestamps
from miv.core.operator import OperatorMixin
from miv.core.wrapper import wrap_cacher
from miv.mea import mea_map


def peri_stimulus_time(spike_list):
    """
    Compute the peri-stimulus time of the given spike train.

    Examples
    --------

    How to draw Peri-stimulus time histogram (ISIH):

        >>> from miv.statistics import peri_stimulus_time
        >>> import matplotlib.pyplot as plt
        >>> pst = peri_stimulus_time(spikestamp)
        >>> plt.hist(pst)

    If one wants to get the bin-count, you can use `numpy.digitize` and `numpy.bincount`:

        >>> import numpy as np
        >>> max_time = spikestamps.max()
        >>> num_bins = 20
        >>> time_interval = np.linspace(0, max_time, num_bins)
        >>> digitized = np.digitize(pst, time_interval)
        >>> bin_counts = np.bincount(digitized)

    Parameters
    ----------
    spikes : SpikestampsType
        Single spike-stamps

    Returns
    -------
        interval: numpy.ndarray

    """

    peri_stimulus_times = np.sum(np.array(spike_list), 0)
    return peri_stimulus_times


@dataclass
class PeristimulusActivity(OperatorMixin):
    # Binning configuration
    # Default: 400ms domain, 4ms binsize
    mea: str = None
    interval: float = 0.4  # seconds
    tag: str = "peri-stimulus activity plot"

    stimulus_length: float = 0.010  # seconds. Skips for removing stimulus artifact

    def __post_init__(self):
        super().__init__()
        if isinstance(self.mea, str):
            self.mea_map = mea_map[self.mea]
        else:
            self.mea_map = mea_map["64_intanRHD"]

    @wrap_cacher("peristimulus activity")
    def __call__(self, events: Spikestamps, spikestamps: Spikestamps):
        # TODO: Change events datatype to be Event, not Spikestamps
        activity = [Spikestamps() for _ in range(spikestamps.number_of_channels)]
        for t_start in events[0]:
            t_end = t_start + self.interval
            view = spikestamps.get_view(
                t_start + self.stimulus_length, t_end + self.stimulus_length
            )
            for channel in range(view.number_of_channels):
                shifted_array = np.asarray(view[channel])
                if shifted_array.size > 0:  # FIXME: Optimize this
                    activity[channel].append(shifted_array - shifted_array[0])
                else:
                    activity[channel].append(shifted_array)
        return activity

    def plot_peristimulus_in_grid_map(self, activity, show=False, save_path=None):
        mea_map = self.mea_map
        nrow, ncol = mea_map.shape
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(ncol * 4, nrow * 4), sharex=True, sharey=True
        )
        for channel in range(len(activity)):
            p = activity[channel]
            if channel not in mea_map:
                continue
            w = np.where(mea_map == channel)
            r = w[0][0]
            c = w[1][0]
            axes[r][c].eventplot(p)
            axes[r][c].set_title(f"channel {channel}")
        # Bottom row
        for i in range(ncol):
            axes[-1, i].set_xlabel("time (s)")

        # Left row
        for i in range(nrow):
            axes[i, 0].set_ylabel("Stimulus Index")

        plt.suptitle("Peri-Stimulus spike activity for each channel")

        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "peristimulus_activity.png"))

        return axes


@dataclass
class PSTH(OperatorMixin):
    # Histogram Configuration
    # Default: 400ms domain, 4ms binsize
    mea: str = None
    binsize: float = 0.004  # seconds
    interval: float = 0.4  # seconds
    tag: str = "peri-stimulus time histogram"

    stimulus_length: float = 0.010  # seconds. Skips for removing stimulus artifact

    def __post_init__(self):
        super().__init__()
        if isinstance(self.mea, str):
            self.mea_map = mea_map[self.mea]
        else:
            self.mea_map = mea_map["64_intanRHD"]

    @wrap_cacher("psth")
    def __call__(self, events: Spikestamps, spikestamps: Spikestamps):
        # TODO: Change events datatype to be Event, not Spikestamps
        n_time = int(np.ceil(self.interval / self.binsize))
        time_axis = np.linspace(0, self.interval, n_time) + self.stimulus_length
        psth = np.zeros((spikestamps.number_of_channels, n_time), dtype=np.float_)
        for t_start in events[0]:
            t_end = t_start + n_time * self.binsize
            bst = spikestamps.binning(
                self.binsize,
                t_start=t_start + self.stimulus_length,
                t_end=t_end + self.stimulus_length,
                return_count=False,
            )
            for channel in range(spikestamps.number_of_channels):
                psth[channel] += bst[channel][:n_time]
        psth /= len(events[0])
        psth /= self.binsize
        return Signal(data=psth.T, timestamps=time_axis, rate=1.0 / self.binsize)

    def plot_psth_in_grid_map(self, psth, show=False, save_path=None):
        mea_map = self.mea_map
        nrow, ncol = mea_map.shape
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(ncol * 4, nrow * 4), sharex=True, sharey=True
        )
        for channel in range(psth.number_of_channels):
            p = psth[channel]
            time = psth.timestamps
            if channel not in mea_map:
                continue
            w = np.where(mea_map == channel)
            r = w[0][0]
            c = w[1][0]
            axes[r][c].plot(time, p)
            axes[r][c].set_title(f"channel {channel}")
        # Bottom row
        for i in range(ncol):
            axes[-1, i].set_xlabel("time (s)")

        # Left row
        for i in range(nrow):
            axes[i, 0].set_ylabel("mean (channels) spike rate per bin")

        plt.suptitle("PSTH: stimulating electrode")

        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "psth.png"))

        return axes


@dataclass
class PSTHOverlay(OperatorMixin):
    # TODO: Experimental
    # Histogram Configuration
    # Default: 400ms domain, 4ms binsize
    mea: str = None
    tag: str = "peri-stimulus time histogram overlay"

    stimulus_length: float = 0.010

    def __post_init__(self):
        super().__init__()
        if isinstance(self.mea, str):
            self.mea_map = mea_map[self.mea]
        else:
            self.mea_map = mea_map["64_intanRHD"]

    def __call__(self, *psths):
        return psths

    def plot_psth_in_grid_map(
        self,
        psths,
        show=False,
        save_path=None,
    ):
        mea_map = self.mea_map
        nrow, ncol = mea_map.shape
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(ncol * 4, nrow * 4), sharex=True, sharey=True
        )
        for idx, psth in enumerate(psths):
            for channel in range(psth.number_of_channels):
                p = psth[channel]
                time = psth.timestamps
                if channel not in mea_map:
                    continue
                w = np.where(mea_map == channel)
                r = w[0][0]
                c = w[1][0]
                axes[r][c].plot(time, p, label=f"PSTH {idx}")
                axes[r][c].set_title(f"channel {channel}")
        # Bottom row
        for i in range(ncol):
            axes[-1, i].set_xlabel("time (s)")

        # Left row
        for i in range(nrow):
            axes[i, 0].set_ylabel("mean spike/bin")

        # Legend
        for i, j in itertools.product(range(nrow), range(ncol)):
            axes[i, j].legend(loc="best")

        plt.suptitle("PSTH: stimulating electrode")

        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "psth.png"))

        return axes

    def plot_psth_area_trend(
        self,
        psths,
        show=False,
        save_path=None,
    ):
        mea_map = self.mea_map
        nrow, ncol = mea_map.shape
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(ncol * 4, nrow * 4), sharex=True, sharey=True
        )
        for channel in range(psths[0].number_of_channels):
            if channel not in mea_map:
                continue
            w = np.where(mea_map == channel)
            r = w[0][0]
            c = w[1][0]

            hist = []
            for idx, psth in enumerate(psths):
                p = psth[channel]
                time = psth.timestamps
                hist.append(np.trapz(p, time))
            axes[r][c].plot(hist)
            axes[r][c].set_title(f"channel {channel}")
        # Bottom row
        for i in range(ncol):
            axes[-1, i].set_xlabel("data points")

        # Left row
        for i in range(nrow):
            axes[i, 0].set_ylabel("Area under PSTH")

        plt.suptitle("PSTH Trend")

        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "psth_trend.png"))

        return axes
