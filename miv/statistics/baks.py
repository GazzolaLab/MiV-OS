__doc__ = """
Bayesian Adaptive Kernel Smoother (BAKS)
BAKS is a method by Ahmandi _[1]_ for estimating progression of firing rate from spiketrain data.
It uses kernel smoothing technique with adaptive bandwidth, based on a Bayesian approach.

References
----------
[1] Ahmadi N, Constandinou TG, Bouganis CS (2018) Estimation of neuronal firing rate using Bayesian Adaptive Kernel Smoother (BAKS). PLOS ONE 13(11): e0206794. https://doi.org/10.1371/journal.pone.0206794
[2] https://github.com/nurahmadi/BAKS
"""
__all__ = [
    "BAKSResult",
    "BayesianAdaptiveKernelSmoother",
    "bayesian_adaptive_kernel_smoother",
]

from dataclasses import dataclass, field
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps
from numba import njit
from tqdm import tqdm

from miv.core import EagerOpNodeBase, Spikestamps


def bayesian_adaptive_kernel_smoother(
    spikestamps, probe_time, alpha=4, beta=None, progress_bar=False
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
    beta : float, optional
        Scale parameter. By default each channel uses ``n_spikes ** (4 / 5)``.

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

        channel_beta = n_spikes ** (4.0 / 5.0) if beta is None else beta
        if channel_beta <= 0:
            raise ValueError("beta must be positive")
        ratio = _numba_ratio_func(probe_time, spiketimes, alpha, channel_beta)
        hs[channel] = (sps.gamma(alpha) / sps.gamma(alpha + 0.5)) * ratio

        firing_rate, firing_rate_for_spike = _numba_firing_rate(
            spiketimes, probe_time, hs[channel]
        )
        firing_rates[channel] = firing_rate
    return hs, firing_rates


@njit(parallel=False)
def _numba_ratio_func(probe_time, spiketimes, alpha, beta):
    # alpha = 1: spike rate contribute up to 1000 sec
    # alpha = 4: spike rate contribute up to 10 sec

    n_time = probe_time.shape[0]
    ratio = np.zeros(n_time)

    for j in range(n_time):
        scale = np.square(probe_time[j] - spiketimes) / 2 + 1.0 / beta
        minimum = np.min(scale)
        normalized = scale / minimum
        numerator = np.sum(normalized ** (-alpha))
        denominator = np.sum(normalized ** (-alpha - 0.5))
        ratio[j] = np.sqrt(minimum) * numerator / denominator
    return ratio


@njit(parallel=False)
def _numba_firing_rate(spiketimes, probe_time, h):
    n_spikes = spiketimes.shape[0]
    n_time = probe_time.shape[0]
    firing_rate = np.zeros((n_time, n_spikes))
    for j in range(n_time):
        for i in range(n_spikes):
            firing_rate[j, i] = (1 / (np.sqrt(2 * np.pi) * h[j])) * np.exp(
                -0.5 * ((probe_time[j] - spiketimes[i]) / h[j]) ** 2
            )
    return firing_rate.sum(axis=1), firing_rate


@dataclass
class BAKSResult:
    probe_times: np.ndarray
    bandwidths: np.ndarray
    firing_rates: np.ndarray
    alpha: float
    beta_rule: str


@dataclass
class BayesianAdaptiveKernelSmoother(EagerOpNodeBase):
    """MiV operator for Bayesian adaptive kernel smoothing."""

    alpha: float = 4.0
    beta: float | None = None
    sample_rate: float = 500.0
    t_start: float | None = None
    t_end: float | None = None
    progress_bar: bool = False
    algorithm_version: str = "rc-kt-baks-v1"
    tag: str = field(default="bayesian adaptive kernel smoother", init=False)

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.sample_rate <= 0:
            raise ValueError("alpha and sample_rate must be positive")
        super().__init__()

    def __call__(self, spikestamps: Spikestamps) -> BAKSResult:
        start = (
            spikestamps.get_first_spikestamp() if self.t_start is None else self.t_start
        )
        end = spikestamps.get_last_spikestamp() if self.t_end is None else self.t_end
        if end <= start:
            raise ValueError("BAKS requires a positive time interval")
        probe_times = np.arange(start, end, 1.0 / self.sample_rate)
        bandwidths, rates = bayesian_adaptive_kernel_smoother(
            spikestamps,
            probe_times,
            alpha=self.alpha,
            beta=self.beta,
            progress_bar=self.progress_bar,
        )
        return BAKSResult(
            probe_times=probe_times,
            bandwidths=bandwidths,
            firing_rates=rates,
            alpha=self.alpha,
            beta_rule="n_spikes**(4/5)" if self.beta is None else str(self.beta),
        )

    def plot_firing_rates(self, result, inputs, show=False, save_path=None):
        fig, axis = plt.subplots(figsize=(9, 4))
        for rate in result.firing_rates:
            axis.plot(result.probe_times, rate, alpha=0.25, linewidth=0.8)
        axis.plot(
            result.probe_times,
            np.nanmean(result.firing_rates, axis=0),
            color="black",
            linewidth=2,
            label="channel mean",
        )
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Firing rate (Hz)")
        axis.set_title("Bayesian adaptive kernel smoothing")
        axis.legend()
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(os.path.join(save_path, "baks_firing_rates.png"), dpi=180)
        if show:
            plt.show()
        plt.close(fig)
