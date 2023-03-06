__all__ = ["PSTH", "peri_stimulus_time"]

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
class PSTH(OperatorMixin):
    # Histogram Configuration
    # Default: 400ms domain, 4ms binsize
    mea: str = None
    binsize: float = 0.004  # seconds
    interval: float = 0.4  # seconds
    tag: str = "peri-stimulus time histogram"

    minimum_stimulus_gap = 0.010
    stimulus_length = 0.010

    def __post_init__(self):
        super().__init__()
        if isinstance(self.mea, str):
            self.mea_map = mea_map[self.mea]
        else:
            self.mea_map = mea_map["64_intanRHD"]

    def __call__(self, events: Spikestamps, spikestamps: Spikestamps):
        n_time = (
            np.rint((self.interval - self.stimulus_length) / (self.binsize)).astype(
                np.int_
            )
            - 1
        )
        time_axis = np.linspace(self.stimulus_length, self.interval, n_time)
        psth = np.zeros((spikestamps.number_of_channels, n_time))
        for channel, spikestamps in enumerate(spikestamps):
            for t_start in events:
                t_end = t_start + self.interval
                bst = spikestamps.get_view(
                    t_start + self.stimulus_length,
                    t_end + self.stimulus_length,
                ).binning(self.bin_size)
                psth[channel] += bst
        psth /= len(events)
        psth /= self.binsize
        return Signal(data=psth, timestamps=time_axis, rate=1.0 / self.binsize)

    def plot_psth_in_grid_map(
        self, psth, show=False, save_path=None, axes=None, label=""
    ):
        mea_map = self.mea_map
        nrow, ncol = mea_map.shape
        if axes is None:
            fig, axes = plt.subplots(
                nrow, ncol, figsize=(nrow * 4, ncol * 4), sharex=True, sharey=True
            )
        for channel, p in range(psth.number_of_channels):
            p = psth[channel]
            time = psth.timestamps
            if channel not in mea_map:
                continue
            w = np.where(mea_map == channel)
            r = w[0][0]
            c = w[1][0]
            axes[r][c].plot(time, p, label=label)
            axes[r][c].set_title(f"channel {channel+1}")
        # Bottom row
        for i in range(ncol):
            axes[-1, i].set_xlabel("time (s)")

        # Left row
        for i in range(nrow):
            axes[i, 0].set_ylabel("mean (channels) spike rate per bin")

        # Legend
        for i, j in itertools.product(range(nrow), range(ncol)):
            axes[i, j].legend()

        plt.suptitle("PSTH: stimulating electrode")

        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "psth.png"))

        return axes
