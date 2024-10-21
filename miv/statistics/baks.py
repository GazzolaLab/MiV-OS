__doc__ = """
Bayesian Adaptive Kernel Smoother (BAKS)
BAKS is a method by Ahmandi _[1]_ for estimating progression of firing rate from spiketrain data.
It uses kernel smoothing technique with adaptive bandwidth, based on a Bayesian approach.

References
----------
[1] Ahmadi N, Constandinou TG, Bouganis CS (2018) Estimation of neuronal firing rate using Bayesian Adaptive Kernel Smoother (BAKS). PLOS ONE 13(11): e0206794. https://doi.org/10.1371/journal.pone.0206794
[2] https://github.com/nurahmadi/BAKS
"""
__all__ = ["bayesian_adaptive_kernel_smoother"]

import time

import numpy as np
import scipy.special as sps
from numba import njit, prange
from tqdm import tqdm

from miv.core.datatype import Spikestamps


def bayesian_adaptive_kernel_smoother(spikestamps, probe_time, alpha=1):
    """
    Bayesian Adaptive Kernel Smoother (BAKS)

    Parameters
    ----------
    spiketimes : Spikestamps
        spike event times
    probe_time : array_like
        time at which the firing rate is estimated. Typically, we assume the number of probe_time is much smaller than the number of spikes events.
    alpha : float, optional
        shape parameter, by default 1

    Returns
    -------
    hs : array_like
        adaptive bandwidth (channels, n_time)
    firing_rates : array_like
        estimated firing rate (channels, n_times)
    """
    num_channels = spikestamps.number_of_channels
    firing_rates = np.zeros((num_channels, len(probe_time)))
    firing_rate_for_spike_list = []
    hs = np.zeros((num_channels, len(probe_time)))
    for channel in range(num_channels):
        spiketimes = np.asarray(spikestamps[channel])
        n_spikes = len(spiketimes)
        beta = n_spikes ** (4./5.)
        if n_spikes == 0:
            continue

        ratio = _numba_ratio_func(probe_time, spiketimes, alpha, beta)
        hs[channel] = (sps.gamma(alpha) / sps.gamma(alpha + 0.5)) * ratio

        firing_rate, firing_rate_for_spike = _numba_firing_rate(
            spiketimes, probe_time, hs[channel]
        )
        firing_rates[channel] = firing_rate
        firing_rate_for_spike_list.append(firing_rate_for_spike)
    return hs, firing_rates, firing_rate_for_spike_list


@njit(parallel=False)
def _numba_ratio_func(probe_time, spiketimes, alpha, beta):
    n_spikes = spiketimes.shape[0]
    n_time = probe_time.shape[0]
    sum_numerator = np.zeros(n_time)
    sum_denominator = np.zeros(n_time)
    for i in range(n_spikes):
        # for j in prange(n_time):
        for j in range(n_time):
            val = ((probe_time[j] - spiketimes[i]) ** 2) / 2 + 1 / beta
            sum_numerator[j] += val ** (-alpha)
            sum_denominator[j] += val ** (-alpha - 0.5)
    ratio = sum_numerator / sum_denominator
    return ratio


@njit(parallel=False)
def _numba_firing_rate(spiketimes, probe_time, h):
    std = 5
    n_spikes = spiketimes.shape[0]
    n_time = probe_time.shape[0]
    firing_rate = np.zeros((n_time, n_spikes))
    # stime = np.searchsorted(spiketimes, probe_time - std * h * 2)
    # etime = np.searchsorted(spiketimes, probe_time + std * h * 2)
    # for j in prange(n_time):

    for j in range(n_time):
        # for i in range(stime[j], etime[j]):
        for i in range(n_spikes):
            firing_rate[j, i] = (1 / (np.sqrt(2 * np.pi) * h[j])) * np.exp(
                -(((probe_time[j] - spiketimes[i]) / (2 * h[j])) ** 2)
            )
    return firing_rate.sum(axis=1), firing_rate


if __name__ == "__main__":
    import sys
    import time

    import matplotlib.pyplot as plt

    from miv.core.datatype import Spikestamps

    # from numba import set_num_threads
    # set_num_threads(4)

    seed = 0
    np.random.seed(seed)

    t = 30
    num_channels = 8
    total_time = 600  # seconds
    N = 80 * total_time  # Number of spikes
    n_evaluation_points = int(total_time)
    evaluation_points = np.linspace(0, total_time, n_evaluation_points)

    spikestamps = [
        np.sort(np.random.random(N)) * total_time for _ in range(num_channels)
    ]
    spikestamps = Spikestamps(spikestamps)

    stime = time.time()
    alpha = 4.0
    hs, firing_rates, firing_rate_for_spike = bayesian_adaptive_kernel_smoother(
        spikestamps, evaluation_points, alpha=alpha
    )
    etime = time.time()
    # print(alpha, beta)

    std = hs[0][0] * 3

    # print(hs)
    # print(firing_rates)

    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    # # plt.plot(evaluation_points, firing_rates[0])
    # ax1.eventplot(spikestamps[0], color="k")
    # ax1.axvline(t, color="r", linestyle="--")
    # ax1.axvspan(t - std, t + std, color="r", alpha=0.5)

    # ax2.plot(spikestamps[0], firing_rate_for_spike[0][0])
    # ax2.axvline(t, color="r", linestyle="--")
    # ax2.axvspan(t - std, t + std, color="r", alpha=0.5)

    # ax3.plot(spikestamps[0], np.cumsum(firing_rate_for_spike[0][0]))
    # ax3.axvline(t, color="r", linestyle="--")
    # ax3.axvspan(t - std, t + std, color="r", alpha=0.5)

    # plt.figure()
    # plt.plot(evaluation_points, firing_rates[0])

    # plt.show()

    print(f"Elapsed time: {etime-stime:.4f} seconds")
