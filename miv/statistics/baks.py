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
from numba import njit, prange
from scipy.special import gamma
from tqdm import tqdm

from miv.core.datatype import Spikestamps


def bayesian_adaptive_kernel_smoother(spikestamps, probe_time, alpha=1, beta=1):
    """
    Bayesian Adaptive Kernel Smoother (BAKS)

    Parameters
    ----------
    spiketimes : Spikestamps
        spike event times
    probe_time : array_like
        time at which the firing rate is estimated
    alpha : float, optional
        shape parameter, by default 1
    beta : float, optional
        scale parameter, by default 1

    Returns
    -------
    hs : array_like
        adaptive bandwidth (channels, n_time)
    firing_rates : array_like
        estimated firing rate (channels, n_times)
    """
    num_channels = spikestamps.number_of_channels
    firing_rates = np.zeros((num_channels, len(probe_time)))
    hs = np.zeros((num_channels, len(probe_time)))
    for channel in range(num_channels):
        spiketimes = np.asarray(spikestamps[channel])
        n_spikes = len(spiketimes)
        if n_spikes == 0:
            continue

        # sum_numerator = 0
        # sum_denominator = 0
        # for i in tqdm(range(n_spikes), position=1, leave=False, desc="get sum_numerator and sum_denominator"):
        #    sum_numerator += (((probe_time - spiketimes[i])**2)/2 + 1/beta)**(-alpha)
        #    sum_denominator += (((probe_time - spiketimes[i])**2)/2 + 1/beta)**(-alpha-0.5)
        # h = (gamma(alpha)/gamma(alpha+0.5))*(sum_numerator/sum_denominator)
        ratio = _numba_ratio_func(probe_time, spiketimes, alpha, beta)
        h = (gamma(alpha) / gamma(alpha + 0.5)) * ratio
        hs[channel] = h

        # firing_rate = np.zeros(len(probe_time))
        # for j in tqdm(range(n_spikes), position=1, leave=False, desc="get firing_rate"):
        #    firing_rate += (1/(np.sqrt(2*np.pi)*h))*np.exp(-((probe_time - spiketimes[j])**2)/(2*h**2))
        firing_rate = _numba_firing_rate(spiketimes, probe_time, h)
        firing_rates[channel] = firing_rate
    return hs, firing_rates


@njit(parallel=True)
def _numba_ratio_func(probe_time, spiketimes, alpha, beta):
    n_spikes = spiketimes.shape[0]
    n_time = probe_time.shape[0]
    sum_numerator = np.zeros(n_time)
    sum_denominator = np.zeros(n_time)
    for i in range(n_spikes):
        for j in prange(n_time):
            val = ((probe_time[j] - spiketimes[i]) ** 2) / 2 + 1 / beta
            sum_numerator[j] += val ** (-alpha)
            sum_denominator[j] += val ** (-alpha - 0.5)
    ratio = sum_numerator / sum_denominator
    return ratio


@njit(parallel=True)
def _numba_firing_rate(spiketimes, probe_time, h):
    n_spikes = spiketimes.shape[0]
    n_time = probe_time.shape[0]
    firing_rate = np.zeros(n_time)
    for i in range(n_spikes):
        for j in prange(n_time):
            firing_rate[j] += (1 / (np.sqrt(2 * np.pi) * h[j])) * np.exp(
                -((probe_time[j] - spiketimes[i]) ** 2) / (2 * h[j] ** 2)
            )
    return firing_rate
