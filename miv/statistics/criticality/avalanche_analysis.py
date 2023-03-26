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
from scipy.optimize import curve_fit

from miv.core.datatype import Spikestamps
from miv.core.operator import OperatorMixin
from miv.core.wrapper import wrap_cacher


@dataclass
class AvalancheAnalysis(OperatorMixin):
    r"""
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
        This influence the threshold of the avalanche size.

    Returns
    -------
    starts_time : np.ndarray
        The start time of each avalanche.
    ends_time : np.ndarray
        The end time of each avalanche.
    durations : np.ndarray
        The duration of each avalanche. (ends_time - starts_time) * bin_size
    size : np.ndarray
        The number of active neurons/channels in each avalanche.
    branching_ratio : np.ndarray
        The branching ratio of each avalanche: the number of neurons active at time step t
        divided by the number of neurons active at time step t-1.

        .. math::
            \sum(thisShape[1:] / thisShape[:-1]) / Avalanche.duration
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
        durations = ends_time - starts_time

        # Compute size and branching ratio
        size = np.zeros(len(starts), dtype=np.int_)
        branching_ratio = np.zeros(len(starts), dtype=np.float_)
        for idx, (s, e) in enumerate(zip(starts, ends)):
            if s == e:
                raise ValueError("Avalanche size is 0")
            avalanche = bincount.data[s:e, :]
            size[idx] = np.count_nonzero(avalanche.sum(axis=0))
            shape = avalanche.sum(axis=1)
            branching_ratio[idx] = np.sum(shape[1:] / shape[:-1]) / durations[idx]

        return starts_time, ends_time, durations, size, branching_ratio

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
        starts_time, ends_time, _, _, _ = outputs

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

    def plot_power_law_fitting(self, outputs, show=False, save_path=None):
        starts_time, ends_time, durations, size, branching_ratio = outputs

        def power(x, a, c):
            return c * (x**a)

        def neg_power(x, a, c):
            return c * (x ** (-a))

        nbins = 100

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        hist, bins = np.histogram(size, bins=nbins)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        axes[0].hist(size, bins=logbins, histtype="step", label="data")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("size (# channels)")
        axes[0].set_ylabel("Event Frequency")
        hist, bins = np.histogram(size, bins=logbins)
        popt, pcov = curve_fit(neg_power, bins[:-1][hist > 1], hist[hist > 1])
        tau = popt[0]
        axes[0].plot(logbins, neg_power(logbins, *popt), label=f"fit {tau=:.2f}")
        axes[0].legend()
        axes[1].set_ylim(bottom=5e-1)

        hist, bins = np.histogram(durations, bins=nbins)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        axes[1].hist(durations, bins=logbins, histtype="step", label="data")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("duration (s)")
        axes[1].set_ylabel("Event Frequency")
        hist, bins = np.histogram(durations, bins=logbins)
        popt, pcov = curve_fit(neg_power, bins[:-1][hist > 1], hist[hist > 1])
        alpha = popt[0]
        axes[1].plot(logbins, neg_power(logbins, *popt), label=f"fit {alpha=:.2f}")
        axes[1].legend()
        axes[1].set_ylim(bottom=5e-1)

        def width(p, w):
            return 10 ** (np.log10(p) + w / 2.0) - 10 ** (np.log10(p) - w / 2.0)

        values = []
        avearges = []
        for value in np.unique(durations):
            positions = np.array([value])
            axes[2].boxplot(
                size[durations == value],
                positions=positions,
                widths=width(positions, 0.1),
                showfliers=False,
            )
            values.append(value)
            avearges.append(np.mean(size[durations == value]))
        axes[2].set_xscale("log")
        axes[2].set_yscale("log")
        axes[2].set_xlabel("duration (s)")
        axes[2].set_ylabel("Average size (# channels)")
        popt, pcov = curve_fit(power, values, avearges)
        axes[2].plot(logbins, power(logbins, *popt), label=f"fit 1/svz={popt[0]:.2f}")
        axes[2].legend()
        axes[2].set_title(f"({(alpha-1)/(tau-1)=:.2f})")

        if save_path is not None:
            plt.savefig(os.path.join(save_path, "avalanche_power_fitting.png"))
        if show:
            plt.show()

    def plot_branching_ratio_histogram(self, outputs, show=False, save_path=None):
        return
        starts_time, ends_time, durations, size, branching_ratio = outputs

        nbins = 100

        fig, ax = plt.subplots(1, 1)
        # hist, bins = np.histogram(size, bins=nbins)
        # logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        ax.hist(
            branching_ratio[np.nonzero(branching_ratio)],
            bins=nbins,
            histtype="step",
            label="data",
        )
        # axes[0].set_xscale("log")
        # axes[0].set_yscale("log")
        ax.set_xlabel("branching ratio")
        ax.set_ylabel("Event Frequency")
        # hist, bins = np.histogram(size, bins=logbins)
        # popt, pcov = curve_fit(neg_power, bins[:-1][hist > 1], hist[hist > 1])
        # tau = popt[0]
        # axes[0].plot(logbins, neg_power(logbins, *popt), label=f"fit {tau=:.2f}")
        # axes[0].legend()
        # axes[0].set_ylim([5e-1, 1e3])
        ax.set_title("branching ratio")
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "branching_ratio.png"))
        if show:
            plt.show()
