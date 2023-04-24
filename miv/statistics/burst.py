__all__ = ["burst"]

import os

import matplotlib.pyplot as plt
import numpy as np

from miv.statistics.spiketrain_statistics import interspike_intervals
from miv.typing import SpikestampsType


def burst(spiketrains: SpikestampsType, channel: int, min_isi: float, min_len: int):
    """
    Calculates parameters critical to characterize bursting phenomenon on a single channel. [1]_
    Bursting is defined as the occurence of a specified number of spikes (usually >10), with a small interspike interval (usually < 100ms) [2]_, [3]_

    Parameters
    ----------
    spikes : SpikestampsType
       Single spike-stamps
    channel : int
       Channel to analyze
    min_isi : float
       Minimum Interspike Interval (in seconds) to be considered as bursting [standard = 0.1]
    min_len : int
       Minimum number of simultaneous spikes to be considered as bursting [standard = 10]

    Returns
    -------
    start_time: float
       The time instances when a burst starts
    burst_duration: float
       The time duration of a particular burst
    burst_len: float
       Number of spikes in a particular burst
    burst_rate: float
       firing rates corresponding to particular bursts

    .. [1] Beggs, John M., and Mark A. Plenz. "Neuronal bursting in cortical circuits." Journal of Neuroscience 23 (2003): 11167-11177.
    .. [2] Chiappalone, Michela, et al. "Burst detection algorithms for the analysis of spatio-temporal patterns
    in cortical networks of neurons." Neurocomputing 65 (2005): 653-662.
    .. [3] Eisenman, Lawrence N., et al. "Quantification of bursting and synchrony in cultured
    hippocampal neurons." Journal of neurophysiology 114.2 (2015): 1059-1071.
    """

    spike_interval = interspike_intervals(spiketrains[channel])
    assert spike_interval.all() > 0, "Inter Spike Interval cannot be zero"
    burst_spike = (spike_interval <= min_isi).astype(
        np.bool_
    )  # Only spikes within specified min ISI are 1 otherwise 0 and are stored

    # Try to find the start and end indices for burst interval
    delta = np.logical_xor(burst_spike[:-1], burst_spike[1:])
    interval = np.where(delta)[0]
    if len(interval) % 2:
        interval = np.append(interval, len(delta))
    interval += 1
    interval = interval.reshape([-1, 2])
    mask = np.diff(interval) >= min_len
    interval = interval[mask.ravel(), :]
    Q = np.array(interval)

    if np.sum(Q) == 0:
        start_time = 0
        end_time = 0
        burst_duration = 0
        burst_rate = 0
        burst_len = 0
    else:
        spike = np.array(spiketrains[channel])
        start_time = spike[Q[:, 0]]
        end_time = spike[Q[:, 1]]
        burst_len = Q[:, 1] - Q[:, 0] + 1
        burst_duration = end_time - start_time
        burst_rate = burst_len / (burst_duration)

    return start_time, burst_duration, burst_len, burst_rate
