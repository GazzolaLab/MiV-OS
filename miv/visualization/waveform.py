__doc__ = """
Module for extracting each spike waveform and visualize.
"""
__all__ = ["extract_waveforms", "plot_waveforms"]

from typing import Any, Optional, Union, Tuples, Dict

import os
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.signal import lfilter, savgol_filter

import matplotlib.pyplot as plt

from miv.typing import SignalType, SpikestampsType

# TODO: Modularize the entire process.


def extract_waveforms(
    signal: SignalType,
    spikes_idx: SpikestampsType,
    sampling_rate: float,
    pre: float = 0.001,
    post: float = 0.002,
    return_spikes_idx: bool = False,
) -> Union[np.ndarray, Tuples[np.ndarray, np.ndarray]]:
    """
    Extract spike waveforms as signal cutouts around each spike index as a spikes x samples numpy array

    Parameters
    ----------
    signal : SignalType
        The signal as a 1-dimensional numpy array
    spikes_idx : SpikestampsType
        The sample index of all spikes as a 1-dim numpy array
    sampling_rate : float
        The sampling frequency in Hz
    pre : float
        The duration of the cutout before the spike in seconds
    post : float
        The duration of the cutout after the spike in seconds
    return_spikes_idx : bool
        If set to True, return spike index that correspond to each cutout. If the spike is
        located at the outer edge of the array, they are not included in this extraction.
        (default=False)

    Returns
    -------
    Stack of spike cutout: np.ndarray or Tuples[np.ndarray, np.ndarray]
        Return stacks of spike cutout; shape(n_spikes, width).
        If return_spikes_idx is set to True, return a tuple of spike cutout and spike index.

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
    cutouts: np.ndarray,
    sampling_rate: float,
    pre: float = 0.001,
    post: float = 0.002,
    n_spikes: Optional[int] = 100,
    color: str = "k",  # TODO: change typing to matplotlib color
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
    plot_kwargs : Dict[Any, Any]
        Addtional keyword-arguments for matplotlib.pyplot.plot.

    Returns
    -------
    Figure : plt.Figure

    """
    if n_spikes is None:
        n_spikes = cutouts.shape[0]
    n_spikes = min(n_spikes, cutouts.shape[0])

    if not plot_kwargs:
        plot_kwargs = {}

    # TODO: Need to match unit
    time_in_us = np.arange(-pre * 1000, post * 1000, 1e3 / sampling_rate)
    fig = plt.figure(figsize=(12, 6))

    for i in range(n_spikes):
        plt.plot(
            time_in_us,
            cutouts[
                i,
            ],
            color,
            linewidth=1,
            alpha=0.3,
            **plot_kwargs
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (uV)")
        plt.title("Cutouts")

    return fig
