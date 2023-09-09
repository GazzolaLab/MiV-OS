__doc__ = """Multi-channel signal and spike rendering for each MEA channels"""
__all__ = ["ReplayRecording"]

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import inspect
import logging
import multiprocessing as mp
import os
import shutil
from dataclasses import dataclass

import matplotlib
import matplotlib.animation as manimation
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from miv.core.datatype import Signal, Spikestamps
from miv.core.operator import OperatorMixin
from miv.core.policy import StrictMPIRunner
from miv.mea.protocol import MEAGeometryProtocol
from miv.typing import SignalType
from miv.visualization.utils import command_run


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
        self.runner = StrictMPIRunner()

    def __call__(
        self,
        signals: Signal,
        signals2: Signal,
        spikestamps: Spikestamps,
        cutouts: Dict[int, Signal],
        mea: MEAGeometryProtocol,
    ):  # pragma: no cover
        comm = self.runner.comm
        rank = self.runner.get_rank()
        size = self.runner.get_size()

        assert inspect.isgenerator(signals)
        assert inspect.isgenerator(signals2)

        for vidx, (signal, signal2) in enumerate(zip(signals, signals2)):
            spikestamps_view = spikestamps.get_view(
                signal.get_start_time(), signal.get_end_time()
            )
            rate = signal.rate

            n_steps_in_window = int(self.window_size * rate)
            interval = int(rate // self.fps * self.play_speed)
            n_steps = int((signal.timestamps.size - n_steps_in_window) // interval)

            # Distribute tasks
            if size > n_steps:
                if rank >= n_steps:
                    logging.warning(f"rank {rank} is idle")
                    continue
                size = n_steps
            tasks = np.arange(n_steps)
            mytask = np.array_split(tasks, size)[rank]

            # Output Images
            plt.rcParams.update({"font.size": 10})

            images_path = os.path.join(self.analysis_path, f"render_{vidx=}")
            if self.runner.is_root():
                os.makedirs(self.analysis_path, exist_ok=True)
                if os.path.exists(images_path):
                    shutil.rmtree(images_path)
                os.makedirs(images_path, exist_ok=True)
            comm.barrier()

            fig = plt.figure(figsize=(16, 20))
            if self.runner.is_root():
                pbar = tqdm(
                    total=len(mytask),
                    desc="Rendering (rank 0)",
                    disable=not self.progress_bar,
                )
            for step in mytask:
                fig.clf()
                outer = gridspec.GridSpec(mea.nrow, mea.ncol, wspace=0.2, hspace=0.2)

                sindex = int(step * interval)
                time = signal.timestamps[sindex : sindex + n_steps_in_window]

                for channel in range(signal.number_of_channels):
                    if channel not in cutouts:
                        continue
                    if channel not in mea.grid:
                        continue

                    loc = mea.get_ixiy(channel)
                    mea_index = loc[0] * mea.ncol + loc[1]
                    inner = gridspec.GridSpecFromSubplotSpec(
                        3,
                        1,
                        subplot_spec=outer[mea_index],
                        wspace=0.1,
                        hspace=0.1,
                        height_ratios=[0.4, 0.4, 0.2],
                    )

                    # Plot signal on the top
                    ax1 = fig.add_subplot(inner[0])
                    ax1.plot(
                        time,
                        signal.data[sindex : sindex + n_steps_in_window, channel],
                    )
                    ax1.set_title(f"Channel {channel}")
                    med = np.median(np.abs(signal.data[:, channel])) * 15
                    y_min = -med
                    y_max = med
                    ax1.set_ylim(y_min, y_max)

                    # Plot signal on the middle
                    ax2 = fig.add_subplot(inner[1], sharex=ax1)
                    ax2.plot(
                        time,
                        signal2.data[sindex : sindex + n_steps_in_window, channel],
                    )
                    med = np.median(np.abs(signal2.data[:, channel])) * 15
                    y_min = -med
                    y_max = med
                    ax2.set_ylim(y_min, y_max)

                    # Plot spikestamps on the bottom
                    ax3 = fig.add_subplot(inner[2], sharex=ax1)
                    ax3.eventplot(
                        spikestamps_view.get_view(time.min(), time.max())[channel]
                    )
                    ax3.set_yticks([])

                    if loc[0] == mea.nrow - 1:
                        ax3.set_xlabel("time (sec)")
                    else:
                        ax3.set_xticks([])
                        ax3.set_xticks([], minor=True)

                fig.tight_layout()
                plt.savefig(
                    os.path.join(images_path, f"{step:05d}.png"),
                    dpi=self.dpi,
                )
                if self.runner.is_root():
                    pbar.update(1)
            if self.runner.is_root():
                pbar.close()
            plt.close(plt.gcf())
            comm.barrier()

            if self.runner.is_root():
                # Concatenate images using ffmpeg
                video_name = os.path.join(self.analysis_path, f"render_{vidx=}.mp4")
                ffmpeg_path = shutil.which("ffmpeg")
                if ffmpeg_path is None:
                    logging.warning("ffmpeg not found")
                else:
                    # "-threads",
                    # f"{mp.cpu_count()}",
                    cmd = [
                        f"{ffmpeg_path}",
                        "-r",
                        f"{self.fps}",
                        "-i",
                        f"{images_path}/%05d.png",
                        "-b:v",
                        "90M",
                        "-vcodec",
                        "mpeg4",
                        f"{video_name}",
                    ]
                    command_run(cmd)

                # if os.path.exists(images_path):
                #    shutil.rmtree(images_path)
            comm.barrier()

        return
