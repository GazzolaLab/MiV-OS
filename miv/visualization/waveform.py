from typing import Optional

import os
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.signal import lfilter, savgol_filter
from scipy import fftpack

# MCS PyData tools
# import McsPy
# import McsPy.McsData
# from McsPy import ureg, Q_

# VISUALIZATION TOOLS
import matplotlib.pyplot as plt


# Spike waveform
def extract_waveforms(
    signal, spikes_idx, sampling_rate, pre=0.001, post=0.002, return_spikes_idx=False
):
    """
    Extract spike waveforms as signal cutouts around each spike index as a spikes x samples numpy array

    :param signal: The signal as a 1-dimensional numpy array
    :param spikes_idx: The sample index of all spikes as a 1-dim numpy array
    :param sampling_rate: The sampling frequency in Hz
    :param pre: The duration of the cutout before the spike in seconds
    :param post: The duration of the cutout after the spike in seconds
    """
    cutouts = []
    pre_idx = int(pre * sampling_rate)
    post_idx = int(post * sampling_rate)
    for index in spikes_idx:
        if index - pre_idx >= 0 and index + post_idx <= signal.shape[0]:
            cutout = signal[(index - pre_idx) : (index + post_idx)]
            cutouts.append(cutout)
    if return_spikes_idx:
        return (
            np.stack(cutouts),
            spikes_idx[
                np.logical_and(
                    spikes_idx - pre_idx >= 0, spikes_idx + post_idx <= signal.shape[0]
                )
            ],
        )
    else:
        return np.stack(cutouts)


def plot_waveforms(
    cutouts,
    sampling_rate,
    pre=0.001,
    post=0.002,
    n_spikes: Optional[int] = 100,
    color="k",
    show=True,
):
    """
    Plot an overlay of spike cutouts

    :param cutouts: A spikes x samples array of cutouts
    :param sampling_rate: The sampling frequency in Hz
    :param pre: The duration of the cutout before the spike in seconds
    :param post: The duration of the cutout after the spike in seconds
    :param n_spikes: The number of cutouts to plot. None to plot all. (Default: 100)
    :param color: The line color as a pyplot line/marker style. Default: 'k'=black
    :param show: Set this to False to disable showing the plot. Default: True
    """
    if n_spikes is None:
        n_spikes = cutouts.shape[0]
    n_spikes = min(n_spikes, cutouts.shape[0])
    time_in_us = np.arange(-pre * 1000, post * 1000, 1e3 / sampling_rate)
    if show:
        plt.figure(figsize=(12, 6))

    for i in range(n_spikes):
        plt.plot(
            time_in_us,
            cutouts[
                i,
            ],
            color,
            linewidth=1,
            alpha=0.3,
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (uV)")
        plt.title("Cutouts")

    if show:
        plt.show()
