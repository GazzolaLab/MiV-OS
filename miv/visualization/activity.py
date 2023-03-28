__all__ = ["NeuralActivity"]

import os
from dataclasses import dataclass

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from miv.core.operator import OperatorMixin
from miv.statistics import spike_counts_with_kernel
from miv.visualization.utils import interp_2d


@dataclass
class NeuralActivity(OperatorMixin):
    bin_size: float = 0.001  # sec
    firing_rate_interval: float = 1.0  # sec
    skip_interval: int = 50  # frames to skip. TODO: refactor

    mea = None
    tag: str = "neural activity render"

    progress_bar: bool = False

    fps: int = 25

    def __post_init__(self):
        super().__init__()

    def __call__(self, spikestamps):
        assert self.mea is not None, "MEA is not set"

        spiketrains_bins = spikestamps.binning(self.bin_size, return_count=True)
        probe_times = spiketrains_bins.timestamps[:: self.skip_interval]
        xs = []
        for i in range(spiketrains_bins.number_of_channels):
            x = spike_counts_with_kernel(
                spikestamps[i],
                probe_times,
                lambda x: np.logical_and(x > 0, x < self.firing_rate_interval).astype(
                    np.float_
                ),
            )
            xs.append(x)
        xs = np.asarray(xs)
        xs_grid = np.zeros_like(self.mea)

        xmax = np.max(xs)
        xmin = np.min(xs)

        # Output Images
        FFMpegWriter = manimation.writers["ffmpeg"]
        metadata = dict(
            title="Movie Test", artist="Matplotlib", comment="Movie support!"
        )
        writer = FFMpegWriter(fps=self.fps, metadata=metadata)

        os.makedirs(self.analysis_path, exist_ok=True)
        video_name = os.path.join(self.analysis_path, "output.mp4")

        fig = plt.figure(figsize=(8, 6))
        with writer.saving(fig, video_name, dpi=200):  # TODO: cut each video into 1min
            for timestep, time in tqdm(
                enumerate(probe_times),
                desc="Rendering",
                total=probe_times.shape[0],
                disable=not self.progress_bar,
            ):
                for channel in range(spiketrains_bins.number_of_channels):
                    xs_grid[self.mea == channel] = xs[channel, timestep]

                fig.clf()
                ax = fig.add_subplot(111)
                X, Y, Z = interp_2d(xs_grid)
                pcm = ax.pcolormesh(X, Y, Z, cmap="Oranges", vmin=xmin, vmax=xmax)
                cbar = fig.colorbar(pcm, ax=ax)
                cbar.ax.set_ylabel(
                    f"activity per {self.firing_rate_interval:.02f} sec", rotation=270
                )

                ax.set_xlabel("channels x-axis")
                ax.set_ylabel("channels y-axis")
                ax.set_title(f"Spatial Neural Activity ({time:.02f} sec)")

                writer.grab_frame()
        plt.close(plt.gcf())

        return
