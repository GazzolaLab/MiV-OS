__all__ = ["pairwise_causality_plot", "spike_triggered_average_plot"]

import os

import matplotlib.pyplot as plt
import numpy as np
from viziphant.spike_train_correlation import plot_corrcoef

from miv.core.datatype import Signal, Spikestamps
from miv.statistics import pairwise_causality
from miv.statistics.spiketrain_statistics import binned_spiketrain
from miv.typing import SignalType, SpikestampsType


def pairwise_causality_plot(signal: SignalType, start: int, end: int):
    """
    Plots pairwise Granger Causality

    Parameters
    ----------
    signal : SignalType
        Input signal
    start : int
        Starting point of the signal
    end : int
        End point of the signal

    Returns
    -------
    figure : matplotlib.pyplot.figure
        Contains subplots for directional causalities for X -> Y and Y -> X,
        instantaneous causality between X,Y, and total causality. X and Y
        represents electrodes
    axes : matplotlib.axes
        axes parameters for plot modification

    See Also
    --------
    miv.statistics.pairwise_causality

    """

    # Causality
    corrcoef_mat = pairwise_causality(signal, start, end)

    # Plotting
    fig, axes = plt.subplots(2, 2)
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6
    )
    plot_corrcoef(corrcoef_mat[0], axes=axes[0, 0])
    plot_corrcoef(corrcoef_mat[1], axes=axes[0, 1])
    plot_corrcoef(corrcoef_mat[2], axes=axes[1, 0])
    plot_corrcoef(corrcoef_mat[3], axes=axes[1, 1])
    axes[0, 0].set_xlabel("Electrode")
    axes[0, 1].set_xlabel("Electrode")
    axes[1, 0].set_xlabel("Electrode")
    axes[1, 1].set_xlabel("Electrode")
    axes[0, 0].set_ylabel("Electrode")
    axes[1, 0].set_ylabel("Electrode")
    axes[0, 1].set_ylabel("Electrode")
    axes[1, 1].set_ylabel("Electrode")
    axes[0, 0].set_title("Directional causality X => Y")
    axes[0, 1].set_title("Directional causality Y => X")
    axes[1, 0].set_title("Instantaneous causality of X,Y")
    axes[1, 1].set_title("Total interdependence of X,Y")

    return fig, axes


def spike_triggered_average_plot(
    signal: Signal,
    channel_x: int,
    spiketrains: Spikestamps,
    channel_y: int,
    sampling_freq: float,
    window_length: int,
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
    sampling_freq : float
        sampling frequency for LFP recordings
    window_length : int
        window length to consider before and after spike

    Returns
    -------
    figure : matplotlib.pyplot.Figure
        matplot figure plotting spike triggered average of channel X in the provided window
    axes : matplotlib.axes.Axes
        axes parameters for plot modification

    """

    # Spike Triggered Average
    dt = 1.0 / sampling_freq
    n = np.shape(signal.data[:, channel_x])[0] / sampling_freq
    assert (
        window_length < np.shape(signal.data[:, channel_x])[0] / 2
    ), "Window cannot be longer than signal length"
    spike = spiketrains[channel_x].get_view(0, n).binning(dt)
    lfp = signal.data[:, channel_x]
    spike_times = np.where(spike == 1)[0]
    spike_len = np.shape(spike_times)[0]
    sta = np.zeros([2 * window_length + 1])

    for i in np.arange(spike_len):
        sta += lfp[spike_times[i] - window_length : spike_times[i] + window_length + 1]

    spike_triggered_average = sta / spike_len
    lags = np.arange(-window_length, window_length + 1) * dt * 1000

    # Plotting
    fig, ax = plt.subplots()
    plt.plot(lags, spike_triggered_average)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (\u03bcV)")

    return fig, ax
