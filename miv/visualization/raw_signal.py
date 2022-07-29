__doc__ = """Multi-channel signal plotting for MEA channels"""
__all__ = ["multi_channel_signal_plot"]

from typing import Any, List, Optional

import matplotlib
import matplotlib.animation as manimation
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from miv.mea.protocol import MEAGeometryProtocol
from miv.typing import SignalType

# matplotlib.use("Agg")  # Must be before importing matplotlib.pyplot or pylab!


def multi_channel_signal_plot(
    signal_list: SignalType,
    mea_geometry: MEAGeometryProtocol,
    start_step: int,
    end_step: int,
    n_steps_in_window: int,
    rendering_fps: int,
    video_name: str,
    max_subplot_in_x: int = 8,
    max_subplot_in_y: int = 8,
    dpi: int = 100,
):
    """
    Plotting recorded neuron signals from each channel of MEA. Subplots for each channel are aligned with position of
    electrical probes on MEA.

    Parameters
    ----------
    signal_list : list
        Contains list  2D numpy.ndarray
        List of channel recordings in time.
    mea_geometry : list
        Contains list of tuples.
        Each tuple contains, channel id, channel y position id and channel x position id on MEA grid.
    start_step : int
        Start step for plotting.
    end_step : int
        End step for plotting.
    n_steps_in_window : int
        Window length for plotting channel signal.
    rendering_fps : int
        Video frame rate
    video_name : str
        Video name
    max_subplot_in_x : int
        (default=8)
    max_subplot_in_y : int
        (default=8)
    dpi : int
        (default=100)

    Returns
    -------

    """
    total_steps = end_step - start_step - n_steps_in_window

    channel_id = []
    xid = []
    yid = []
    for i, channel_info in enumerate(mea_geometry):
        channel_id.append(channel_info[0])
        yid.append(channel_info[1])
        xid.append(channel_info[2])
    n_channels = len(channel_id)

    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=rendering_fps, metadata=metadata)
    fig = plt.figure(2, figsize=(20, 12), frameon=True, dpi=dpi)
    plt.rcParams.update({"font.size": 10})
    axs = []

    for y, x in zip(yid, xid):
        axs.append(plt.subplot2grid((max_subplot_in_y, max_subplot_in_x), (y, x)))

    signal_line_list = [None for _ in range(n_channels)]

    for signal_id in range(n_channels):
        signal = signal_list[:, signal_id]

        # x_value = timestamps[start_step : start_step + n_steps_in_window]
        y_value = signal[start_step : start_step + n_steps_in_window]
        signal_line_list[signal_id] = axs[signal_id].plot(y_value, "-", linewidth=3)[0]

        y_min = np.min(signal[start_step:end_step])
        y_max = np.max(signal[start_step:end_step])

        axs[signal_id].set_ylim(y_min, y_max)

    plt.tight_layout()
    fig.align_ylabels()

    with writer.saving(fig, video_name, dpi):
        for step in tqdm(range(total_steps)):
            current_step = start_step + step
            for signal_id in range(n_channels):
                signal = signal_list[:, signal_id]
                # x_value = signal[current_step : current_step + n_steps_in_window, 0]
                y_value = signal[current_step : current_step + n_steps_in_window]

                # signal_line_list[signal_id].set_xdata(x_value)
                signal_line_list[signal_id].set_ydata(y_value)

                # X limits should move together with window
                # axs[signal_id].set_xlim(x_value[0], x_value[-1])

            writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())
