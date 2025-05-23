import matplotlib.pyplot as plt
import numpy as np

from miv.core.datatype import Spikestamps
from miv.statistics import burst
from miv.typing import SpikestampsType


def plot_spiketrain_raster(
    spikestamps: Spikestamps, t_start: float, t_stop: float
):  # pragma: no cover
    """
    Plot spike train in raster

    Parameters
    ----------
    spikestamps : Spikestamps
        Spike stamps
    t_start : float
        Start time
    t_stop : float
        Stop time
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    spikes = spikestamps.get_view(t_start, t_stop)
    print(
        f"Plotting raster plot from {t_start} to {t_stop}: {sum(spikes.get_count())} spikes"
    )
    ax.eventplot(spikes)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(f"Raster plot (from {t_start} to {t_stop})")
    ax.set_xlim(t_start, t_stop)
    return fig, ax


def plot_burst(
    spiketrains: SpikestampsType, min_isi: float, min_len: float
):  # pragma: no cover
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
        ax.broken_barh(a, (i, 0.5), facecolors="tab:orange")

    times = np.concatenate([train for train in spiketrains])
    channel = np.concatenate(
        [np.full_like(train, i, dtype=np.float_) for i, train in enumerate(spiketrains)]
    )
    ax.scatter(times, channel, s=2, marker="o", color="blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Electrode")

    return fig, ax
