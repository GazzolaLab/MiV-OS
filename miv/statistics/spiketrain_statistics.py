__all__ = [
    "firing_rates",
    "MFRComparison",
    "interspike_intervals",
    "coefficient_variation",
    "peri_stimulus_time",
    "binned_spiketrain",  # Deprecated: Remove in the future version ^0.3.0
    "fano_factor",
    "decay_spike_counts",
    "spike_counts_with_kernel",
]

from typing import Any, Dict, Iterable, List, Optional, Union

import datetime
import os
from dataclasses import dataclass

import elephant.statistics
import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq
import scipy
import scipy.signal

from miv.core.datatype import Spikestamps
from miv.core.operator import OperatorMixin
from miv.core.wrapper import miv_function
from miv.typing import SpikestampsType


def firing_rates(spiketrains: Spikestamps) -> Dict[str, Any]:
    """
    Process basic spiketrains statistics: rates, mean, variance.

    Parameters
    ----------
    spiketrains : Iterable[neo.core.SpikeTrain]

    Returns
    -------
    Iterable[Any]

    """
    rates = []
    for spikestamp in spiketrains.neo():
        if len(spikestamp) == 0:
            rates.append(0)
            continue
        mfr = elephant.statistics.mean_firing_rate(spikestamp)
        if isinstance(mfr, pq.quantity.Quantity):
            mfr = mfr.magnitude[()]
        rates.append(mfr)

    rates_mean_over_channel = np.mean(rates)
    rates_variance_over_channel = np.var(rates)
    return {
        "rates": rates,
        "mean": rates_mean_over_channel,
        "variance": rates_variance_over_channel,
    }


@dataclass
class MFRComparison(OperatorMixin):
    recording_duration: float = None
    tag: str = "Mean Firing Rate Comparison"

    def __call__(self, pre_spiketrains: Spikestamps, post_spiketrains: Spikestamps):
        pre_rates = firing_rates(pre_spiketrains)["rates"]
        post_rates = firing_rates(post_spiketrains)["rates"]
        if self.recording_duration is None:
            self.recording_duration = max(
                pre_spiketrains.get_last_spikestamp(),
                post_spiketrains.get_last_spikestamp(),
            ) - min(
                pre_spiketrains.get_first_spikestamp(),
                post_spiketrains.get_first_spikestamp(),
            )
        return pre_rates, post_rates

    def __post_init__(self):
        super().__init__()

    def plot_mfr_comparison(self, output, show=False, save_path=None):
        MFR_pre, MFR_post, duration = output

        MFR = np.geomspace(1e-1, 1e2)
        kl = 7
        ku = 7
        sigma = (
            np.sqrt(np.mean([MFR_pre, MFR_post]) * self.recording_duration)
            / self.recording_duration
        )

        fig = plt.figure()
        plt.plot(MFR, MFR, "b--")
        plt.plot(MFR, MFR - kl * sigma, "b")
        plt.plot(MFR, MFR + ku * sigma, "b")
        plt.scatter(MFR_pre, MFR_post)
        ax = plt.gca()
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim([1e-1, 1e2])
        ax.set_ylim([1e-1, 1e2])
        ax.set_xlabel("MFR pre")
        ax.set_ylabel("MFR post")

        if show:
            plt.show()
        if save_path is not None:
            fig.savefig(os.path.join(f"{save_path}", "mfr.png"))

        return ax


def interspike_intervals(spikes: SpikestampsType):
    """
    Compute the inter-spike intervals of the given spike train.

    Examples
    --------

    How to draw Inter-spike interval histogram (ISIH):

        >>> from miv.statistics import interspike_intervals
        >>> import matplotlib.pyplot as plt
        >>> interspike_interval = interspike_intervals(spikestamp)
        >>> plt.hist(interspike_interval)

    If one wants to get the bin-count, you can use `numpy.digitize` and `numpy.bincount`:

        >>> import numpy as np
        >>> max_time = spikestamps.max()
        >>> num_bins = 20
        >>> time_interval = np.linspace(0, max_time, num_bins)
        >>> digitized = np.digitize(interspike_interval, time_interval)
        >>> bin_counts = np.bincount(digitized)

    Parameters
    ----------
    spikes : SpikestampsType
        Single spike-stamps

    Returns
    -------
        interval: numpy.ndarray

    """

    spike_interval = np.diff(spikes)
    return spike_interval


def coefficient_variation(self, spikes: SpikestampsType):
    """
    The Coefficient of variation: ratio of interspike standard deviation over mean.

    Parameters
    ----------
    spikes : SpikestampsType
        Single spike-stamps

    Returns
    -------
        coefficient variation : float
    """
    interspike = self.interspike_intervals()
    return np.std(interspike) / np.mean(interspike)


def peri_stimulus_time(spike_list: List[SpikestampsType]):
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


def fano_factor(
    spiketrains: SpikestampsType,
    channel: float,  # TODO: the function should be independent of channel.
    t_start: float,
    t_end: float,
    n_bins: float,
):
    """
    Calculates the Fano factor for given signal by dividing it into the specified number of bins

    Parameters
    ----------
    spiketrains : SpikestampsType
        Single spike-stamps
    channel : float
        electrode/channel
    t_start : float
        Binning start time
    t_end : float
        Binning end time
    n_bins : float
        Number of bins

    Returns
    -------
        fano_fac: float
        fanofactor for the specified channel and conditions

    """
    raise NotImplementedError("This function is not implemented yet.")
    assert (
        t_start < t_end
    ), f"End time {t_end} cannot be smaller or equal to start time {t_start}."
    assert n_bins > 0, "Number of bins should be a positive integer"
    bin_spike = spiketrains.get_view(t_start, t_end).binning(bin_size=0.002)
    assert np.sum(bin_spike) != 0, "The channel has no spikes"
    large_bin = []
    bin_length = np.int32(np.size(bin_spike) / n_bins)
    count = 0
    for i in range(n_bins):
        large_bin.append(np.sum(bin_spike[count : count + bin_length]))
        count += bin_length
    bin_array = np.array(large_bin)
    fano_fac = np.var(bin_array) / np.mean(bin_array)

    return fano_fac


def decay_spike_counts(
    spiketrain, probe_times, amplitude=1.0, decay_rate=5, batchsize=256
):
    """
    decay_spike_counts.

    Both spiketrain and probe_times should be a 1-d array representing time.

    Parameters
    ----------
    spiketrain :
        spiketrain
    probe_times :
        probe_times
    amplitude :
        amplitude
    decay_rate :
        decay_rate
    batchsize :
        batchsize
    """
    if len(spiketrain) == 0:
        return np.zeros_like(probe_times)
    batchsize = min(spiketrain.shape[0], batchsize)
    num_sections = spiketrain.shape[0] // batchsize + (
        1 if spiketrain.shape[0] % batchsize > 0 else 0
    )

    result = np.zeros_like(probe_times)
    for subspiketrain in np.array_split(spiketrain, num_sections):
        exponent = (
            np.tile(probe_times, (len(subspiketrain), 1)) - subspiketrain[:, None]
        )
        mask = exponent < 0
        exponent[mask] = 0
        decay_count = np.exp(-decay_rate * exponent)
        decay_count[mask] = 0
        result += decay_count.sum(axis=0)
    return result


def spike_counts_with_kernel(spiketrain, probe_times, kernel, batchsize=32):
    if len(spiketrain) == 0:
        return np.zeros_like(probe_times)

    batchsize = min(spiketrain.shape[0], batchsize)
    num_sections = spiketrain.shape[0] // batchsize + (
        1 if spiketrain.shape[0] % batchsize > 0 else 0
    )

    result = np.zeros_like(probe_times)
    for subspiketrain in np.array_split(spiketrain, num_sections):
        exponent = (
            np.tile(probe_times, (len(subspiketrain), 1)) - subspiketrain[:, None]
        )
        mask = exponent < 0
        exponent[mask] = 0
        decay_count = kernel(exponent)
        # decay_count = np.exp(-decay_rate * exponent)
        decay_count[mask] = 0
        result += decay_count.sum(axis=0)
    return result


def binned_spiketrain(self):
    raise DeprecationWarning("Use `timestamps.binning` instead.")
