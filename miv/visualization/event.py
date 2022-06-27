import os

import matplotlib.pyplot as plt
import numpy as np

from miv.statistics import burst
from miv.typing import SpikestampsType


def plot_burst(spiketrains: SpikestampsType, min_isi: float, min_len: float):
    """
    Plots burst events across electrodes  to characterize bursting phenomenon on a singl channel

    Parameters
    ----------
    spikes : SpikestampsType
           Single spike-stamps
    min_isi : float
       Minimum Interspike Interval (in seconds) to be considered as bursting [standard = 0.1]
    min_len : float
       Minimum number of simultaneous spikes to be considered as bursting [standard = 10]

    Returns
    -------
    figure, axes
       matplot figure with bursts plotted for all electrodes
    """

    fig, ax = plt.subplots()
    start_time = []
    burst_duration = []
    burst_len = []
    burst_rate = []
    for i in np.arange(len(spiketrains)):
        start_time, burst_duration, burst_len, burst_rate = burst(
            spiketrains, i, min_isi, min_len
        )
        a = np.column_stack((start_time, burst_duration))
        ax.broken_barh(a, (i + 1, 0.5), facecolors="tab:orange")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Electrode")

    return fig, ax
