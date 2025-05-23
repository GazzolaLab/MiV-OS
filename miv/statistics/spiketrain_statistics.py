__all__ = [
    "firing_rates",
    "MFRComparison",
    "interspike_intervals",
    "coefficient_variation",
    "binned_spiketrain",
    "fano_factor",
    "spike_counts_with_kernel",
    "decay_spike_counts",
    "instantaneous_spike_rate",
]

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq

from miv.core.datatype import Spikestamps
from miv.core.operator.operator import OperatorMixin
from miv.typing import SpikestampsType


def firing_rates(spiketrains: Spikestamps) -> dict[str, Any]:
    """
    Process basic spiketrains statistics: rates, mean, variance.

    Parameters
    ----------
    spiketrains : Iterable[neo.core.SpikeTrain]

    Returns
    -------
    Iterable[Any]

    """
    import elephant.statistics

    rates = []
    if sum(spiketrains.get_count()) == 0:
        return {
            "rates": [0.0 for _ in range(spiketrains.number_of_channels)],
            "mean": 0.0,
            "variance": 0.0,
        }
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
    channels: list[int] = None

    def __call__(self, pre_spiketrains: Spikestamps, post_spiketrains: Spikestamps):
        assert (
            pre_spiketrains.number_of_channels == post_spiketrains.number_of_channels
        ), (
            f"Number of channels does not match: {pre_spiketrains.number_of_channels} vs {post_spiketrains.number_of_channels}"
        )
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

    def plot_mfr_comparison(self, output, inputs, show=False, save_path=None):
        MFR_pre, MFR_post = output

        if self.channels is not None:
            MFR_pre = MFR_pre[self.channels]
            MFR_post = MFR_post[self.channels]

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


def fano_factor(
    spiketrains: SpikestampsType,
    bin_size: float = 0.002,
    t_start: float | None = None,
    t_end: float | None = None,
):
    """
    Calculates the Fano factor for given signal by dividing it into the specified number of bins

    Parameters
    ----------
    spiketrains : SpikestampsType
        Single spike-stamps
    bin_size : int
        Size of the bin in second. Default is 2ms
    t_start : float
        Binning start time
    t_end : float
        Binning end time

    Returns
    -------
        fano_fac: float
        fanofactor for the specified channel and conditions

    """
    bin_spike = spiketrains.binning(
        bin_size=bin_size, t_start=t_start, t_end=t_end, return_count=True
    )
    fano_factor = np.zeros(bin_spike.number_of_channels, dtype=np.float64)
    for channel in range(bin_spike.number_of_channels):
        array = bin_spike[channel]
        if np.sum(array) == 0:
            fano_factor[channel] = np.nan
        fano_factor[channel] = np.var(array) / np.mean(array)
    return fano_factor


# TODO: combine the class of spike counting functions with various kernels.
def spike_counts_with_kernel(spiketrain, probe_times, kernel: Callable, batchsize=32):
    """
    Both spiketrain and probe_times should be a 1-d array representing time.

    Parameters
    ----------
    spiketrain :
        Single-channel spiketrain
    probe_times :
        probe_times
    kernel :
        kernel function
    batchsize :
        batchsize
    """
    if len(spiketrain) == 0:
        # Zero spike in the channel
        return np.zeros_like(probe_times)
    spiketrain = np.asarray(spiketrain)
    result = np.zeros_like(probe_times)

    # Non-batch implementation : temporary
    for s in spiketrain:
        mask = (s + 1e-8) > probe_times
        exponent = kernel((s - probe_times)[mask])
        exponent[exponent < 1e-2] = 0.0
        result[mask] += exponent
    return result

    # Batch implementation
    batchsize = min(spiketrain.shape[0], batchsize)
    num_sections = spiketrain.shape[0] // batchsize + (
        1 if spiketrain.shape[0] % batchsize > 0 else 0
    )

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


def decay_spike_counts(
    spiketrain, probe_times, amplitude=1.0, decay_rate=5, batchsize=256
):
    def kernel(x):
        return amplitude * np.exp(-decay_rate * x) / decay_rate

    return spike_counts_with_kernel(spiketrain, probe_times, kernel, batchsize)


def instantaneous_spike_rate(spiketrain, probe_times, window=1, batchsize=32):
    """
    Set window=1 for unit Hz.
    """

    def kernel(x):
        return np.logical_and(x >= 0, x <= window).astype(np.float64)

    return spike_counts_with_kernel(spiketrain, probe_times, kernel, batchsize)


def binned_spiketrain(self):
    raise DeprecationWarning("Use `timestamps.binning` instead.")
