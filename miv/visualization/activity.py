__all__ = ["NeuralActivity"]

import logging
import os
from dataclasses import dataclass

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from miv.core.datatype import Spikestamps
from miv.core.operator import OperatorMixin
from miv.core.policy import StrictMPIRunner
from miv.mea import MEAGeometryProtocol
from miv.statistics.spiketrain_statistics import spike_counts_with_kernel
from miv.visualization.utils import interp_2d


@dataclass
class NeuralActivity(OperatorMixin):
    bin_size: float = 0.001  # sec
    firing_rate_interval: float = 1.0  # sec
    skip_interval: int = 50  # frames to skip. TODO: refactor

    tag: str = "neural activity render"

    progress_bar: bool = False

    fps: int = 25
    minimum_split_size: int = 60  # sec. Each video is at least 1 min

    def __post_init__(self):
        super().__init__()
        self.runner = StrictMPIRunner()

    def __call__(self, spikestamps: Spikestamps, mea: MEAGeometryProtocol):
        comm = self.runner.comm
        rank = self.runner.get_rank()
        size = self.runner.get_size()

        # Binning in first rank
        if self.runner.is_root():
            spiketrains_bins = spikestamps.binning(self.bin_size, return_count=True)
            probe_times = spiketrains_bins.timestamps[:: self.skip_interval]
        else:
            spiketrains_bins = None
            probe_times = None
        spiketrains_bins = comm.bcast(spiketrains_bins, root=self.runner.get_root())
        probe_times = comm.bcast(probe_times, root=self.runner.get_root())

        # Split Tasks
        num_frames = probe_times.shape[0]
        if size * self.fps * self.minimum_split_size > num_frames:
            size = np.ceil(num_frames / (self.fps * self.minimum_split_size)).astype(
                np.int_
            )
            if rank == self.runner.get_root():
                logging.warning(
                    f"Too many ranks. Splitting into {size} tasks for total {num_frames} frames."
                )
            if rank >= size:
                return

        probe_times = np.array_split(probe_times, size)[rank]
        start_time = probe_times[0]
        end_time = probe_times[-1]
        logging.info(
            f"{rank=} | rendering from {start_time=:.03f} to {end_time=:.03f}: {probe_times.shape[0]} frames."
        )

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

        xmax = np.max(xs)
        xmin = np.min(xs)

        # Output Images
        FFMpegWriter = manimation.writers["ffmpeg"]
        metadata = dict(
            title="Movie Test", artist="Matplotlib", comment="Movie support!"
        )
        writer = FFMpegWriter(fps=self.fps, metadata=metadata)

        os.makedirs(self.analysis_path, exist_ok=True)
        video_name = os.path.join(
            self.analysis_path, f"output_from_{start_time:.03f}_to_{end_time:.03f}.mp4"
        )

        fig = plt.figure(figsize=(8, 6))
        with writer.saving(fig, video_name, dpi=200):  # TODO: cut each video into 1min
            for timestep, time in tqdm(
                enumerate(probe_times),
                desc=f"Rendering {rank}/{size}",
                position=rank,
                total=probe_times.shape[0],
                disable=not self.progress_bar,
            ):
                X, Y, Z = mea.map_data(xs[:, timestep])

                fig.clf()
                ax = fig.add_subplot(111)
                # X, Y, Z = interp_2d(Z)
                ax.plot(mea.coordinates[:, 0], mea.coordinates[:, 1], "k.", ms=1)
                pcm = ax.pcolormesh(
                    X, Y, Z, cmap="Oranges", vmin=xmin, vmax=xmax, shading="gouraud"
                )
                cbar = fig.colorbar(pcm, ax=ax)
                cbar.ax.set_ylabel(
                    f"activity per {self.firing_rate_interval:.03f} sec", rotation=270
                )

                ax.set_aspect("equal")
                ax.set_xlabel("channels x-axis")
                ax.set_ylabel("channels y-axis")
                ax.set_title(f"Spatial Neural Activity ({time:.03f} sec)")

                writer.grab_frame()
        plt.close(plt.gcf())

        return
