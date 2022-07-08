__all__ = [
    "firing_rates",
    "interspike_intervals",
    "coefficient_variation",
    "peri_stimulus_time",
    "binned_spiketrain",
    "fano_factor",
]

from typing import Any, Dict, Iterable, List, Optional, Union

import datetime

import elephant.statistics
import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq
import scipy
import scipy.signal

from miv.typing import SpikestampsType


# FIXME: For now, we provide the free function for simple usage. For more
# advanced statistical analysis, we should have a module wrapper.
def firing_rates(
    spiketrains: Union[pq.Quantity, Iterable[neo.core.SpikeTrain]],
    # t_start: Optional[float] = None,
    # t_stop: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Process basic spiketrains statistics: rates, mean, variance.

    Parameters
    ----------
    spiketrains : Iterable[neo.core.SpikeTrain]

    Returns
    -------
    Iterable[Any]

    """
    rates = [
        float(elephant.statistics.mean_firing_rate(spikestamp).magnitude)
        for spikestamp in spiketrains
    ]
    rates_mean_over_channel = np.mean(rates)
    rates_variance_over_channel = np.var(rates)
    return {
        "rates": rates,
        "mean": rates_mean_over_channel,
        "variance": rates_variance_over_channel,
    }


def interspike_intervals(spikes: SpikestampsType):
    """
    Compute the inter-spike intervals of the given spike train.

    Examples
    --------

    How to draw Inter-spike interval histogram (ISIH):

        >>> from miv.statistics import inter_spike_intervals
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


def binned_spiketrain(
    spiketrains: SpikestampsType,
    channel: float,
    t_start: float,
    t_end: float,
    bin_size: float,
):
    """
    Forms a binned spiketrain using the spiketrain


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
    bin_size : float
        bin size in seconds

    Returns
    -------
        bin_spike: numpy.ndarray
        binned spiketrain with 1 corresponding to spike and zero otherwise

    """

    n_bins = int((t_end - t_start) / bin_size + 1)
    time = np.linspace(t_start, bin_size * (n_bins - 1), n_bins)
    bin_spike = np.zeros(n_bins)
    spike = spiketrains[channel].magnitude
    bins = np.digitize(spike, time)
    bin_spike[bins - 1] = 1

    return bin_spike


def fano_factor(
    spiketrains: SpikestampsType,
    channel: float,
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

    bin_spike = binned_spiketrain(spiketrains, channel, t_start, t_end, 0.002)
    assert np.sum(bin_spike) != 0, "The channel has no spikes"
    large_bin = []
    bin_length = np.int32(np.size(bin_spike) / n_bins)
    count = 0
    for i in np.arange(n_bins):
        large_bin.append(np.sum(bin_spike[count : count + bin_length]))
        count += bin_length
    bin_array = np.array(large_bin)
    fano_fac = np.var(bin_array) / np.mean(bin_array)

    return fano_fac
