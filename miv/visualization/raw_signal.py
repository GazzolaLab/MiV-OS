__doc__ = """Multi-channel signal plotting for MEA channels"""
__all__ = ["MultiChannelSignalVisualization"]

from typing import Any, List, Optional

import inspect
import os
from dataclasses import dataclass

import matplotlib
import matplotlib.animation as manimation
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from miv.core.datatype import Spikestamps
from miv.core.operator import OperatorMixin
from miv.mea.protocol import MEAGeometryProtocol
from miv.typing import SignalType
from miv.visualization.utils import interp_2d


@dataclass
class MultiChannelSignalVisualization(OperatorMixin):
    average_interval: int = 50  # frames to skip. TODO: refactor

    tag: str = "signal render"

    progress_bar: bool = False

    fps: int = 25
    dpi: int = 200

    def __post_init__(self):
        super().__init__()

    def __call__(self, signals: Spikestamps, mea: MEAGeometryProtocol):

        if not inspect.isgenerator(signals):
            signals = [signals]

        for vidx, signal in enumerate(signals):
            probe_times = signal.timestamps[:: self.average_interval]
            n = (signal.data.shape[0] // self.average_interval) * self.average_interval
            xs = np.average(
                signal.data[:n, :].reshape(
                    -1, self.average_interval, signal.number_of_channels
                ),
                axis=1,
            )
            med = np.median(np.abs(xs))
            xmax = 15.0 * med
            xmin = -15.0 * med

            # Output Images
            FFMpegWriter = manimation.writers["ffmpeg"]
            metadata = dict(
                title="Movie Test", artist="Matplotlib", comment="Movie support!"
            )
            writer = FFMpegWriter(fps=self.fps, metadata=metadata)

            os.makedirs(self.analysis_path, exist_ok=True)
            video_name = os.path.join(self.analysis_path, f"output_{vidx}.mp4")

            fig = plt.figure(figsize=(8, 6))
            with writer.saving(fig, video_name, dpi=self.dpi):
                for timestep in tqdm(
                    range(xs.shape[0]),
                    desc="Rendering",
                    total=xs.shape[0],
                    disable=not self.progress_bar,
                ):
                    time = probe_times[timestep]
                    X, Y, Z = mea.map_data(xs[timestep, :])

                    fig.clf()
                    ax = fig.add_subplot(111)
                    # X, Y, Z = interp_2d(Z)
                    ax.plot(mea.coordinates[:, 0], mea.coordinates[:, 1], "k.", ms=1)
                    pcm = ax.pcolormesh(
                        X,
                        Y,
                        Z,
                        cmap="MGBlueOrange",
                        vmin=xmin,
                        vmax=xmax,
                        shading="gouraud",
                    )
                    cbar = fig.colorbar(pcm, ax=ax)
                    cbar.ax.set_ylabel(
                        f"signal averaged over {self.average_interval/signal.rate:.02f} sec",
                        rotation=270,
                    )

                    ax.set_aspect("equal")
                    ax.set_xlabel("channels x-axis (µm)")
                    ax.set_ylabel("channels y-axis (µm)")
                    ax.set_title(f"Signal ({time:.05f} sec)")
                    ax.invert_yaxis()

                    writer.grab_frame()
            plt.close(plt.gcf())

        return
