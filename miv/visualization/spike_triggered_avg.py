import os

import matplotlib.pyplot as plt
import numpy as np

from miv.statistics.spiketrain_statistics import binned_spiketrain
from miv.typing import SignalType, SpikestampsType


def spike_triggered_avg(
    signal: SignalType,
    channel_x: float,
    spiketrains: SpikestampsType,
    channel_y: float,
    sampling_freq: float,
    win: float,
):
    """
    Plots the spike-triggered average of Local Field Potential (LFP) from channel X
    corresponding to spiketrain from channel Y. The spiketrain from channel Y
    can be replaced with stimulation signal to understand stimulus dependent
    LFP on channel X, but take care in providing stimualtion as SpikestampsType.

    Parameters
    ----------
    signal : SignalType
        LFP signal recorded from the electrodes
    channel_x : float
        Channel to consider for LFP data
    spiketrains : SpikestampsType
        Single spike-stamps
    channel_y : float
        Channel to consider for spiketrain data
    sampling_freq: float
        sampling frequency for LFP recordings
    win: float
        window length to consider before and after spike

    Returns
    -------
    figure, axes
       matplot figure plotting spike triggered average of channel X in the provided window
    """

    dt = 1 / sampling_freq
    n = np.shape(signal[:, channel_x])[0] / sampling_freq
    assert (
        win < np.shape(signal[:, channel_x])[0] / 2
    ), "Window cannot be longer than signal length"
    spike = binned_spiketrain(spiketrains, channel_x, 0, n, dt)
    lfp = signal[:, channel_x]
    spike_times = np.where(spike == 1)[0]
    spike_len = np.shape(spike_times)[0]
    sta = np.zeros([2 * win + 1])

    for i in np.arange(spike_len):
        sta += lfp[spike_times[i] - win : spike_times[i] + win + 1]

    spike_triggered_average = sta / spike_len
    lags = np.arange(-win, win + 1) * dt * 1000
    fig, ax = plt.subplots()
    plt.plot(lags, spike_triggered_average)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (\u03bcV)")

    return fig, ax
