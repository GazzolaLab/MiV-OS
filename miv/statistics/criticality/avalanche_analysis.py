__doc__ = """
Avalanche Analysis
"""

__all__ = ["AvalancheAnalysis"]

from typing import List, Optional, Tuple, Union

import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq

from miv.core.datatype import Spikestamps
from miv.core.operator import OperatorMixin
from miv.core.wrapper import wrap_cacher


@dataclass
class AvalancheAnalysis(OperatorMixin):
    """
    An operator for finding avalanches in spike trains.
    It returns the size, duration, and avalanches themselves, defined as all the times each neuron spiked in a given avalanche.

    Parameters
    ----------
    bin_size : float
        The size of the bins in seconds. Default is 0.004 seconds.
    threshold : float
        The threshold for the avalanche size. Default is None.
    time_difference : float
        Minimum time difference between avalanches in seconds.
        Default is equal to the bin_size.
    allow_multiple_spike_per_bin : bool
        If True, allow multiple spikes per bin. Default is False.

    Returns
    -------
    str_avas - a cell array containing all avalanches
    szes - the size of each avalanche in number of spikes
    lens - the duration of each avalanch
    rast - a spike raster in row major ordering over the time range
    specified, if that was specified
    starts - times when avalanches begin

    """

    bin_size: float = 0.004  # in seconds

    # Miscellaneous configurations
    threshold: Optional[float] = None
    time_difference: Optional[float] = None
    allow_multiple_spike_per_bin: bool = False

    tag: str = "Avalanche and Criticality Analysis"

    @wrap_cacher("avalanche_detection")
    def __call__(self, spikestamps: Spikestamps):
        bincount = spikestamps.binning(
            bin_size=self.bin_size, return_count=self.allow_multiple_spike_per_bin
        )
        population_firing = bincount.data.sum(
            axis=bincount._CHANNELAXIS
        )  # Spike count accross channel per bin
        if self.threshold is None:
            self.threshold = (
                population_firing[np.nonzero(population_firing)].mean() / 2.0
            )
        # TODO: try to reuse the code from miv.statistics.burst.burst
        events = (population_firing > self.threshold).astype(np.bool_)

        # pad and find where avalanches start or end based on where 0s change to
        # 1s and vice versa
        concat_events = np.concatenate([[False], events, [False]])
        diff_events = concat_events[1:].astype(np.int_) - concat_events[:-1].astype(
            np.int_
        )
        starts = np.where(diff_events == 1)[0]
        ends = np.where(diff_events == -1)[0]

        if not np.isclose(self.bin_size, self.time_difference):
            starts2 = starts[1:]
            ends2 = ends[:-1]
            avalanche_interval = (starts2 - ends2) * self.bin_size
            inds = np.arange(len(avalanche_interval))
            inds = inds[avalanche_interval > self.time_difference]

            starts = starts[np.concatenate([[0], inds + 1])]
            ends = ends[np.concatenate([inds, [len(ends) - 1]])]

        starts_time = starts * self.bin_size + bincount.timestamps[0]
        ends_time = ends * self.bin_size + bincount.timestamps[0]
        avalanche_lengths = ends_time - starts_time

        return starts_time, ends_time, avalanche_lengths

    def __post_init__(self):
        super().__init__()
        if isinstance(self.bin_size, pq.Quantity):
            self.bin_size = self.bin_size.rescale("s").magnitude
        if isinstance(self.time_difference, pq.Quantity):
            self.time_difference = self.time_difference.rescale("s").magnitude
        if self.time_difference is None:
            self.time_difference = self.bin_size

    def plot_avalanche_on_raster(self, outputs, show=False, save_path=None):
        """Plot firing rate histogram"""
        spikestamps = self.receive()[0]
        starts_time, ends_time, avalanche_lengths = outputs

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.eventplot(spikestamps)

        for start, end in zip(starts_time, ends_time):
            ax.axvspan(start, end, facecolor="r", alpha=0.5)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Channel")
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "raster_avalanche_overlay.png"))
        if show:
            plt.show()
        return ax
