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
from numba import njit

from miv.core.datatype import Spikestamps


def bayesian_adaptive_kernel_smoother(
    spikestamps, probe_time, alpha=4, progress_bar=False
):
    """
    Bayesian Adaptive Kernel Smoother (BAKS)

    Parameters
    ----------
    spiketimes : Spikestamps
        spike event times
    probe_time : array_like
        time at which the firing rate is estimated. Typically, we assume the number of probe_time is much smaller than the number of spikes events.
    alpha : float, optional
        shape parameter, by default 4

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
    for channel in tqdm(
        range(num_channels), desc="Channel: ", disable=not progress_bar
    ):
        spiketimes = np.asarray(spikestamps[channel])
        n_spikes = len(spiketimes)
        if n_spikes == 0:
            continue

        ratio = _numba_ratio_func(probe_time, spiketimes, alpha)
        hs[channel] = (sps.gamma(alpha) / sps.gamma(alpha + 0.5)) * ratio

        firing_rate, firing_rate_for_spike = _numba_firing_rate(
            spiketimes, probe_time, hs[channel]
        )
        firing_rates[channel] = firing_rate
    return hs, firing_rates


@njit(parallel=False)
def _numba_ratio_func(probe_time, spiketimes, alpha):
    # alpha = 1: spike rate contribute up to 1000 sec
    # alpha = 4: spike rate contribute up to 10 sec

    n_spikes = spiketimes.shape[0]
    n_time = probe_time.shape[0]
    sum_numerator = np.zeros(n_time)
    sum_denominator = np.zeros(n_time)

    diff_lim = 10 ** (4.5 / (alpha + 0.5))
    spike_start_indices = np.searchsorted(spiketimes, probe_time - diff_lim)
    spike_end_indices = np.searchsorted(spiketimes, probe_time + diff_lim)

    for j in range(n_time):
        # for i in range(n_spikes):
        _spiketimes = spiketimes[spike_start_indices[j] : spike_end_indices[j]]

        val = (np.square(probe_time[j] - _spiketimes) / 2) ** (-alpha)
        # print(val.shape, val.min(), val.max())
        sum_numerator[j] = val.sum()
        val = (np.square(probe_time[j] - _spiketimes) / 2) ** (-alpha - 0.5)
        sum_denominator[j] = val.sum()

        # for i in range(spike_start_indices[j], spike_end_indices[j]):
        #     val = ((probe_time[j] - spiketimes[i]) ** 2) / 2
        #     sum_numerator[j] += val ** (-alpha)
        #     sum_denominator[j] += val ** (-alpha - 0.5)
    ratio = sum_numerator / (sum_denominator + 1e-14)
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
    import time

    from miv.core.datatype import Spikestamps

    from numba import set_num_threads

    set_num_threads(4)

    seed = 0
    np.random.seed(seed)

    def original_imple(a, L=5):
        num = a ** (-L)
        den = a ** (-L - 0.5)
        ratio = num.sum() / den.sum()
        return ratio

    def stable_ratio(a, L=5):
        # Find the maximum element in a to normalize
        a_min = np.min(a)

        # Compute the normalized terms
        normalized_numerator_terms = (a / a_min) ** (-L)
        normalized_denominator_terms = (a / a_min) ** (-L - 0.5)

        # Calculate the sums
        numerator = np.sum(normalized_numerator_terms)
        denominator = np.sum(normalized_denominator_terms)

        # Compute the final stable ratio
        ratio = np.sqrt(a_min) * (numerator / denominator)

        return ratio

    # Test cases
    test_cases = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1e10, 1e5, 1e3]),
        np.array([0.1, 0.5, 0.9]),
        np.array([1.5, 2.5, 3.5]),
        np.array([1e-3, 1e-5, 1e-10]),
        np.geomspace(1e-40, 1e40, 100),
    ]

    # Run the test cases and display results
    for a in test_cases:
        prev_out = original_imple(a)
        stable_out = stable_ratio(a)
        print(prev_out, stable_out)
    sys.exit()

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
    hs, firing_rates = bayesian_adaptive_kernel_smoother(
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

    print(f"Elapsed time: {etime - stime:.4f} seconds")
