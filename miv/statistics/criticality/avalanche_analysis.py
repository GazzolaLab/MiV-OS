__doc__ = """
Avalanche Analysis
"""

__all__ = ["AvalancheDetection", "AvalancheAnalysis"]

from typing import List, Optional, Tuple, Union

import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from scipy.optimize import curve_fit
from tqdm import tqdm

from miv.core.datatype import Signal, Spikestamps
from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call


@dataclass
class AvalancheDetection(OperatorMixin):
    r"""
    An operator for finding avalanches in spike trains.
    It returns the size, duration, and avalanches themselves, defined as all the times each neuron spiked in a given avalanche.

    Parameters
    ----------
    bin_size : float
        The size of the bins in seconds. Default is 0.004 seconds.
    threshold_percentage : float
        The threshold to detect avalanche.
        Default is 2.0, which means the threshold is the mean of the population firing rate divided by 2.
        Higher the value, less avalanches will be detected, and the avalanche size will be smaller.
    time_difference : float
        Minimum time difference between avalanches in seconds.
        Default is equal to the bin_size.
    allow_multiple_spike_per_bin : bool
        If True, allow multiple spikes per bin. Default is False.
        This influence the threshold of the avalanche size.
    minimum_bins_in_avalanche : int
        The minimum number of bins in an avalanche. Default is 10.
    min_interburst_interval_bound : float
        The minimum interburst interval in seconds. If burst interval is shorter, two burst will be coalaced. Default is 0.1.

    Returns
    -------
    starts : np.ndarray
        The start timestamp of each avalanche.
    ends : np.ndarray
        The end timestamp of each avalanche.
    bincount: Signal
        The binned spike count.

        .. math::
            \sum(thisShape[1:] / thisShape[:-1]) / Avalanche.duration
    """

    bin_size: float = 0.002  # in seconds

    # Miscellaneous configurations
    threshold_percentage: float = 2.0
    time_difference: Optional[float] = None
    allow_multiple_spike_per_bin: bool = False
    minimum_bins_in_avalanche: int = 10

    min_interburst_interval_bound: float = 0.1  # sec
    pre_burst_extension: float = 0.0
    post_burst_extension: float = 0.0

    tag: str = "avalanche detection"

    @cache_call
    def __call__(self, spikestamps: Spikestamps):
        bincount = spikestamps.binning(
            bin_size=self.bin_size, return_count=self.allow_multiple_spike_per_bin
        )
        population_firing = bincount.data.sum(
            axis=bincount._CHANNELAXIS
        )  # Spike count accross channel per bin
        threshold = (
            population_firing[np.nonzero(population_firing)].mean()
            / self.threshold_percentage
        )
        # TODO: try to reuse the code from miv.statistics.burst.burst
        events = (population_firing > threshold).astype(np.bool_)

        # pad and find where avalanches start or end based on where 0s change to
        # 1s and vice versa
        concat_events = np.concatenate([[False], events, [False]])
        diff_events = concat_events[1:].astype(np.int_) - concat_events[:-1].astype(
            np.int_
        )
        starts = np.where(diff_events == 1)[0]
        ends = np.where(diff_events == -1)[0]

        if len(starts) == 0 or len(ends) == 0:
            return np.array([]), np.array([]), bincount

        # remove avalanches that are too short
        if self.minimum_bins_in_avalanche > 1:
            inds = np.where(ends - starts >= self.minimum_bins_in_avalanche)[0]
            starts = starts[inds]
            ends = ends[inds]

        # TODO: duplicated lines
        if len(starts) == 0 or len(ends) == 0:
            return np.array([]), np.array([]), bincount

        if not np.isclose(self.bin_size, self.time_difference):
            starts2 = starts[1:]
            ends2 = ends[:-1]
            avalanche_interval = (starts2 - ends2) * self.bin_size
            inds = np.arange(len(avalanche_interval))
            inds = inds[avalanche_interval > self.time_difference]

            starts = starts[np.concatenate([[0], inds + 1])]
            ends = ends[np.concatenate([inds, [len(ends) - 1]])]

        # Include residual windows
        starts = starts - int(self.pre_burst_extension / self.bin_size)
        ends = ends + int(self.post_burst_extension / self.bin_size)

        # Coalace overlapped intervals
        while True:
            coalace_index = np.where(
                starts[1:]
                <= ends[:-1] + self.min_interburst_interval_bound / self.bin_size
            )[0]
            if coalace_index.size == 0:
                break
            starts = np.delete(starts, coalace_index + 1)
            ends = np.delete(ends, coalace_index)

        return starts, ends, bincount

    def __post_init__(self):
        super().__init__()
        if isinstance(self.bin_size, pq.Quantity):
            self.bin_size = self.bin_size.rescale("s").magnitude
        if isinstance(self.time_difference, pq.Quantity):
            self.time_difference = self.time_difference.rescale("s").magnitude
        if self.time_difference is None:
            self.time_difference = self.bin_size

    def plot_avalanche_on_raster(self, outputs, inputs, show=False, save_path=None):
        """Plot firing rate histogram"""
        if show:
            logging.warning(
                "Plotting avalanche on raster is not supported to be shown interactively."
            )
        spikestamps = self.receive()[0]
        starts, ends, bincount = outputs
        starts_time = starts * self.bin_size + bincount.timestamps[0]
        ends_time = ends * self.bin_size + bincount.timestamps[0]

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.eventplot(spikestamps)

        for start, end in zip(starts_time, ends_time):
            ax.axvspan(start, end, facecolor="r", alpha=0.5)

        ax.set_title("Avalanche Marks on Raster")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Channel")

        # Save
        interval = 10  # sec TODO
        left, right = (
            spikestamps.get_first_spikestamp(),
            spikestamps.get_last_spikestamp(),
        )
        left -= interval * 0.1
        right += interval * 0.1
        for i in range(int(np.ceil((right - left) / interval))):
            ax.set_xlim(left + i * interval, left + (i + 1) * interval)
            if save_path is not None:
                plt.savefig(
                    os.path.join(save_path, f"raster_avalanche_overlay_{i}.png")
                )

            if i > 10:  # For early finish
                break
        plt.close(fig)


@dataclass
class AvalancheAnalysis(OperatorMixin):
    """
    Avalanche and Criticality Analysis

    .. example::

        >>> from miv.statistics.criticality import AvalancheDetection, AvalancheAnalysis

        >>> # Avalanche detection
        >>> avalanche_detection = AvalancheDetection()
        >>> avalanche_analysis = AvalancheAnalysis()
        >>> avalanche_detection >> avalanche_analysis

    Parameters
    ----------
    tag: str
        The tag for the operator. Default is "Avalanche and Criticality Analysis".
    """

    tag: str = "avalanche criticality analysis"

    progress_bar: bool = False

    @cache_call
    def __call__(self, inputs):
        starts, ends, bincount = inputs
        bin_size = 1.0 / bincount.rate
        durations = (ends - starts) * bin_size

        # Compute size and branching ratio
        size = np.zeros(starts.size, dtype=np.int_)
        branching_ratio = np.zeros(starts.size, dtype=np.float_)
        avalanches = []
        for idx, (s, e) in tqdm(
            enumerate(zip(starts, ends)),
            total=len(starts),
            desc="Avalanche analysis",
            disable=not self.progress_bar,
        ):
            if s == e:
                raise ValueError("Avalanche size is 0")
            avalanche = bincount.data[s:e, :]
            avalanches.append(avalanche)
            size[idx] = np.count_nonzero(avalanche.sum(axis=0))
            shape = avalanche.sum(axis=1)
            if np.any(np.isclose(shape[:-1], 0)) or np.isclose(size[idx], 0):
                branching_ratio[idx] = 0.0
            else:
                branching_ratio[idx] = np.sum(shape[1:] / shape[:-1]) / size[idx]

        return durations, size, branching_ratio, avalanches, bin_size

    def __post_init__(self):
        super().__init__()

    def plot_power_law_fitting(self, outputs, inputs, show=False, save_path=None):
        durations, size, branching_ratio, _, _ = outputs

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
        if hist[hist > 1].size > 0:
            try:
                popt, pcov = curve_fit(neg_power, bins[:-1][hist > 1], hist[hist > 1])
                tau = popt[0]
                axes[0].plot(
                    logbins, neg_power(logbins, *popt), label=f"fit {tau=:.2f}"
                )
            except RuntimeError:
                tau = 0
                logging.warning("Power-fit failed. No fitted line will be plotted.")
            except TypeError:
                tau = 0
                logging.warning("Power-fit failed. No fitted line will be plotted.")
            self.tau = tau
        else:
            tau = 0.0
            self.tau = 0.0
        axes[0].legend()
        axes[0].set_ylim(bottom=5e-1)

        hist, bins = np.histogram(durations, bins=nbins)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        axes[1].hist(durations, bins=logbins, histtype="step", label="data")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("duration (s)")
        axes[1].set_ylabel("Event Frequency")
        hist, bins = np.histogram(durations, bins=logbins)
        try:
            if (hist > 1).sum() > 0:
                popt, pcov = curve_fit(neg_power, bins[:-1][hist > 1], hist[hist > 1])
                alpha = popt[0]
                axes[1].plot(
                    logbins, neg_power(logbins, *popt), label=f"fit {alpha=:.2f}"
                )
            else:
                alpha = 0
        except RuntimeError:
            alpha = 0
            logging.warning("Power-fit failed. No fitted line will be plotted.")
        except TypeError:
            alpha = 0
            logging.warning("Power-fit failed. No fitted line will be plotted.")
        self.alpha = alpha
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
        try:
            popt, pcov = curve_fit(power, values, avearges)
            self.svz = popt[0]
            axes[2].plot(
                logbins, power(logbins, *popt), label=f"fit 1/svz={popt[0]:.2f}"
            )
        except RuntimeError:
            logging.warning("Power-fit failed. No fitted line will be plotted.")
            self.svz = 0
        except TypeError:
            logging.warning("Power-fit failed. No fitted line will be plotted.")
            self.svz = 0
        axes[2].legend()
        axes[2].set_title(f"({(alpha-1)/(tau-1)=:.2f})")
        self.svz_estim_ratio = (alpha - 1) / (tau - 1)

        if save_path is not None:
            plt.savefig(os.path.join(save_path, "avalanche_power_fitting.png"))
        if show:
            plt.show()
        plt.close(fig)

    def plot_branching_ratio_histogram(
        self, outputs, inputs, show=False, save_path=None
    ):
        _, _, branching_ratio, _, _ = outputs

        nbins = 100

        fig, ax = plt.subplots(1, 1)
        # hist, bins = np.histogram(size, bins=nbins)
        # logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        self.mean_branching_ratio = branching_ratio[np.nonzero(branching_ratio)].mean()
        ax.hist(
            branching_ratio[np.nonzero(branching_ratio)],
            bins=nbins,
            histtype="step",
            label="data",
        )
        ax.set_xlabel("branching ratio")
        ax.set_ylabel("Event Frequency")
        ax.set_title("branching ratio")
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "branching_ratio.png"))
        if show:
            plt.show()
        plt.close(fig)

    def plot_avalanche_shape_collapse(
        self, outputs, inputs, show=False, save_path=None
    ):
        _, _, _, avalanches, bin_size = outputs

        shapes = defaultdict(list)
        for avalanche in avalanches:
            count = avalanche.shape[0]
            if count <= 1:
                continue
            shapes[count].append(avalanche.sum(axis=1))

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for idx, count in enumerate(shapes.keys()):
            shapes[count] = np.array(shapes[count])
            time = np.arange(count) * bin_size
            T = count * bin_size
            mean = np.array(shapes[count]).mean(axis=0)
            err = np.array(shapes[count]).std(axis=0)
            axes[0].errorbar(
                time, mean, yerr=err, label=f"duration {bin_size*count*1000:.2f}ms"
            )
            axes[1].plot(
                time / T,
                mean / (T ** (self.svz - 1)),
                label=f"duration {bin_size*count*1000:.2f}ms",
            )

        axes[0].set_xlabel("Time in Avalanche (s)")
        axes[0].set_ylabel("Average Number of Firing s(t,T)")
        axes[0].set_title("Raw Shapes")
        # axes[0].legend()
        axes[1].set_xlabel("Time in Avalanche Duration (s / T)")
        axes[1].set_ylabel("Average Number of Firing s(t,T) / T^1/svz-1 ")
        axes[1].set_title(f"Normalized/Collapsed Avalanche Shape {self.svz:.2f}")
        # axes[1].legend()

        if save_path is not None:
            plt.savefig(os.path.join(save_path, "avalanche_shape_collapse.png"))
        if show:
            plt.show()
        plt.close(fig)
