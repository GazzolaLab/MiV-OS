__doc__ = """Multi-channel signal and spike rendering for each MEA channels"""
__all__ = ["ReplayRecording"]

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import inspect
import os
from dataclasses import dataclass

import matplotlib
import matplotlib.animation as manimation
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from miv.core.datatype import Signal, Spikestamps
from miv.core.operator import OperatorMixin
from miv.mea.protocol import MEAGeometryProtocol
from miv.typing import SignalType


@dataclass
class ReplayRecording(OperatorMixin):
    """
    Collective rendering of electrophyiological signals from MEA channels.
    """

    window_size: float = 3.0  # sec. TODO: refactor
    play_speed: float = 1.0

    tag: str = "experiment replay"

    progress_bar: bool = False

    fps: int = 30
    dpi: int = 200

    def __post_init__(self):
        super().__init__()

    def __call__(
        self,
        signals: Signal,
        spikestamps: Spikestamps,
        cutouts: Dict[int, Signal],
        mea: MEAGeometryProtocol,
    ):  # pragma: no cover

        if not inspect.isgenerator(signals):
            signals = [signals]

        for vidx, signal in enumerate(signals):
            spikestamps_view = spikestamps.get_view(
                signal.get_start_time(), signal.get_end_time()
            )
            rate = signal.rate

            n_steps_in_window = int(self.window_size * rate)
            interval = int(rate // self.fps * self.play_speed)
            n_steps = int((signal.timestamps.size - n_steps_in_window) // interval)

            # Output Images
            FFMpegWriter = manimation.writers["ffmpeg"]
            metadata = dict(
                title="Movie Test", artist="Matplotlib", comment="Movie support!"
            )
            writer = FFMpegWriter(fps=self.fps, metadata=metadata)

            plt.rcParams.update({"font.size": 10})

            os.makedirs(self.analysis_path, exist_ok=True)
            video_name = os.path.join(self.analysis_path, f"output_{vidx}.mp4")

            fig = plt.figure(figsize=(16, 16))
            with writer.saving(fig, video_name, dpi=self.dpi):
                for step in tqdm(
                    range(n_steps),
                    desc="Rendering",
                    total=n_steps,
                    disable=not self.progress_bar,
                ):
                    fig.clf()
                    outer = gridspec.GridSpec(
                        mea.nrow, mea.ncol, wspace=0.2, hspace=0.2
                    )
                    for channel in range(signal.number_of_channels):
                        if channel not in cutouts:
                            continue
                        if channel not in mea.grid:
                            continue

                        loc = mea.get_ixiy(channel)
                        mea_index = loc[0] * mea.nrow + loc[1]
                        sindex = int(step * interval)
                        time = signal.timestamps[sindex : sindex + n_steps_in_window]
                        inner = gridspec.GridSpecFromSubplotSpec(
                            2, 1, subplot_spec=outer[mea_index], wspace=0.1, hspace=0.1
                        )

                        # Plot signal on the top
                        ax = fig.add_subplot(inner[0])
                        ax.plot(
                            time,
                            signal.data[sindex : sindex + n_steps_in_window, channel],
                        )
                        ax.set_title(f"Channel {channel}")
                        y_min = np.min(signal.data[:, channel])
                        y_max = np.max(signal.data[:, channel])
                        ax.set_ylim(y_min, y_max)

                        # Plot spikestamps on the bottom
                        ax = fig.add_subplot(inner[1], sharex=ax)
                        ax.eventplot(
                            spikestamps_view.get_view(time.min(), time.max())[channel]
                        )
                        ax.set_xlabel("time (sec)")

                    writer.grab_frame()
            plt.close(plt.gcf())

        return
