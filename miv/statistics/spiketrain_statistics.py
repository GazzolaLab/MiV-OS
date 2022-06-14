__all__ = ["firing_rates", "inter_spike_intervals"]

from typing import Any, Dict, Iterable, Optional, Union

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


def inter_spike_intervals(spikes):
    """
    Compute the inter-spike intervals of the given spike train.

    Examples
    --------

    How to draw Inter-spike interval histogram (ISIH)

        >>> from miv.statistics import inter_spike_intervals
        >>> import matplotlib.pyplot as plt
        >>> interval = inter_spike_intervals(spikestamps)
        >>> plt.hist(interval)

    Parameters
    ----------
    spikes : SpikestampsType

    Returns
    -------
        numpy.ndarray

    """

    return np.diff(spikes)
