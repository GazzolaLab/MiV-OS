__doc__ = """

Spike Train Statistic Tools
===========================

SpikeTrainStatistics
--------------------

.. automethod:: miv.statistics.spike_train_statistics.inter_spike_intervals

"""

__all__ =["inter_spike_intervals"]
import numpy as np


def inter_spike_intervals(spikes):
    """
    Compute the inter-spike intervals of the given spike train.

    Parameters
    ----------
    spikes : numpy.ndarray

    Returns
    -------
        numpy.ndarray

    """


    return np.diff(spikes)
