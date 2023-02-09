__doc__ = """
Module for extracting each spike waveform and visualize.
"""
__all__ = ["extract_waveforms", "plot_waveforms"]

from typing import Any, Dict, Optional, Tuple, Union

import os

import matplotlib
import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq
from scipy.signal import lfilter, savgol_filter
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from miv.signal.spike.cutout import ChannelSpikeCutout, SpikeCutout
from miv.typing import SignalType, SpikestampsType

# TODO: Modularize the entire process.


def extract_waveforms(
    signal: SignalType,
    spikestamps: SpikestampsType,
    channel: Optional[int],
    sampling_rate: float,
    pre: pq.Quantity = 0.001 * pq.s,
    post: pq.Quantity = 0.002 * pq.s,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract spike waveforms as signal cutouts around each spike index as a spikes x samples numpy array

    Parameters
    ----------
    signal : SignalType
        The signal as a 2-dimensional numpy array (length, num_channel)
    spikestamps : SpikestampsType
        The sample index of all spikes as a 1-dim numpy array
    channel : Optional[int]
        Interested channel. If None, assume signal and spikestamps are single channel.
    sampling_rate : float
        The sampling frequency in Hz
    pre : pq.Quantity
        The duration of the cutout before the spike in seconds. (default=0.001 s)
    post : pq.Quantity
        The duration of the cutout after the spike in seconds. (default=0.002 s)

    Returns
    -------
    Stack of spike cutout: np.ndarray
        Return stacks of spike cutout; shape(n_spikes, width).

    """
    if channel is not None:
        signal = signal[:, channel]
        spikestamps = spikestamps[channel]

    cutouts = []
    pre_idx = int(pre * sampling_rate)
    post_idx = int(post * sampling_rate)

    assert (
        pre_idx + post_idx > 0
    ), "Set larger pre/post duration. pre+post duration must be more than 1/sampling_rate."

    # Padding signal
    signal = np.pad(signal, ((pre_idx, post_idx),), constant_values=0)
    for time in spikestamps:
        index = int(round(time * sampling_rate))
        if index >= signal.shape[0] or index + post_idx + pre_idx >= signal.shape[0]:
            raise IndexError(
                "The width of the spike exceeded the signal. "
                "Either timestamp exceeded the maximum time recorded, or "
                "post duration is too large."
            )
        # if index - pre_idx >= 0 and index + post_idx <= signal.shape[0]:
        #    cutout = signal[(index - pre_idx) : (index + post_idx)]
        #    cutouts.append(cutout)
        cutout = signal[index : (index + post_idx + pre_idx)]
        cutouts.append(cutout)

    return np.stack(cutouts)


def plot_waveforms(
    cutouts: np.ndarray,
    sampling_rate: float,
    pre: float = 0.001,
    post: float = 0.002,
    n_spikes: Optional[int] = 100,
    color: str = "k",  # TODO: change typing to matplotlib color
    ax: Optional[matplotlib.axes.Axes] = None,
    return_time: bool = False,
    plot_kwargs: Dict[Any, Any] = None,
) -> plt.Figure:
    """
    Plot an overlay of spike cutouts

    Parameters
    ----------
    cutouts : np.ndarray
        A spikes x samples array of cutouts
    sampling_rate : float
        The sampling frequency in Hz
    pre : float
        The duration of the cutout before the spike in seconds
    post : float
        The duration of the cutout after the spike in seconds
    n_spikes : Optional[int]
        The number of cutouts to plot. None to plot all. (Default: 100)
    color : str
        The line color as a pyplot line/marker style. (Default: 'k'=black)
    ax : Optional[matplotlib.axes.Axes]
        Use provided axes to plot. If none, create new figure.
    plot_kwargs : Dict[Any, Any]
        Addtional keyword-arguments for matplotlib.pyplot.plot.
    """
    if n_spikes is None:
        n_spikes = cutouts.shape[0]
    n_spikes = min(n_spikes, cutouts.shape[0])

    if not plot_kwargs:
        plot_kwargs = {"alpha": 0.3, "linewidth": 1}

    # TODO: Need to match unit
    time = np.arange(-pre * 1000, post * 1000, 1e3 / sampling_rate)

    fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    for i in range(n_spikes):
        ax.plot(
            time,
            cutouts[
                i,
            ],
            color,
            **plot_kwargs,
        )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (uV)")
    ax.set_title("Cutouts")
    if return_time:
        return fig, time
    return fig
