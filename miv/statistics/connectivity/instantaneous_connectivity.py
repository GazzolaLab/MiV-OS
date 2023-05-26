__doc__ = """
Connectivity module for localized instantaneous analysis
"""
__all__ = ["InstantaneousConnectivity"]

from typing import Any, List, Optional, Union

import csv
import functools
import gc
import glob
import itertools
import logging
import multiprocessing as mp
import os
import pathlib
import pickle as pkl
from dataclasses import dataclass

import matplotlib
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyinform
import pyinform.transferentropy as pyte
import quantities as pq
import scipy.stats as spst
from tqdm import tqdm

from miv.core.datatype import Signal, Spikestamps
from miv.core.operator import OperatorMixin
from miv.core.policy import StrictMPIRunner
from miv.core.wrapper import wrap_cacher
from miv.mea import mea_map


@dataclass
class InstantaneousConnectivity(OperatorMixin):
    """ """

    channels: Optional[List[int]] = None
    bin_size: float = 0.001
    tag: str = "instantaneous connectivity rendering"
    progress_bar: bool = False

    te_history: int = 21

    fps: int = 25

    def __post_init__(self):
        super().__init__()
        self.runner = StrictMPIRunner()
        if self.runner.is_root:
            os.makedirs(self.analysis_path, exist_ok=True)
        self.runner.comm.Barrier()

    def __call__(self, spikestamps: Spikestamps) -> np.ndarray:
        """__call__.

        Parameters
        ----------
        spikestamps : Spikestamps
        """
        # comm = self.runner.comm
        rank = self.runner.get_rank()
        size = self.runner.get_size()

        binned_spiketrain = spikestamps.binning(bin_size=self.bin_size)
        n_timesteps = binned_spiketrain.shape[binned_spiketrain._SIGNALAXIS]

        # Channel Selection
        if self.channels is None:
            n_nodes = binned_spiketrain.number_of_channels
            channels = tuple(range(n_nodes))
        else:
            n_nodes = len(self.channels)
            channels = tuple(self.channels)

        # Define task
        pairs = [
            (i, j) for i, j in itertools.product(channels, channels) if i != j and i < j
        ]
        ntasks = len(pairs)

        # Split tasks
        if size > ntasks:
            logging.warning(
                f"Too many ranks. {size-ntasks} number of ranks will be idle."
            )
        tasks = np.array_split(pairs, size)[rank]

        # Run tasks
        disable_tqdm = not self.progress_bar or not self.runner.is_root
        for i, j in tqdm(tasks, disable=disable_tqdm):
            # compute i -> j
            ij_te = self._compute_local_directionality(
                binned_spiketrain[i],
                binned_spiketrain[j],
                self.te_history,
            )
            # compute j -> i
            ji_te = self._compute_local_directionality(
                binned_spiketrain[j],
                binned_spiketrain[i],
                self.te_history,
            )

            FFMpegWriter = manimation.writers["ffmpeg"]
            metadata = dict(
                title="Movie Test", artist="Matplotlib", comment="Movie support!"
            )
            writer = FFMpegWriter(fps=self.fps, metadata=metadata)
            video_name = os.path.join(
                self.analysis_path, f"local_te_channel_{i}_{j}.mp4"
            )

            # plot 3x1 figure: imshow of binnec spiketrain for i and j, ij_te, ji_te
            fig = plt.figure(figsize=(16, 5))
            sidx = 0
            dframe = int(1 / self.bin_size / self.fps)
            with writer.saving(fig, video_name, dpi=200):
                while sidx + dframe < n_timesteps:
                    stime = binned_spiketrain.get_start_time() + sidx * self.bin_size
                    etime = stime + dframe * self.bin_size
                    time = binned_spiketrain.timestamps[sidx : sidx + dframe]

                    fig.clf()
                    ax1 = fig.add_subplot(311)
                    _lineoffsets = [0, 1]
                    _linelengths = [1, 1]
                    ax1.eventplot(
                        spikestamps.select([i, j], keepdims=False).view(stime, etime),
                        lineoffsets=_lineoffsets,
                        linelengths=_linelengths,
                    )
                    ax1.set_yticklabels(["i", "j"])

                    ax2 = fig.add_subplot(312)
                    ax2.plot(time, ij_te[sidx : sidx + dframe])
                    ax2.set_ylabel("i -> j")

                    ax3 = fig.add_subplot(313)
                    ax3.plot(time, ji_te[sidx : sidx + dframe])
                    ax3.set_ylabel("j -> i")

                    ax2.sharex(ax1)
                    ax3.sharex(ax1)

                    # Make it tight layout
                    fig.tight_layout()

                    sidx += dframe

    def _compute_local_directionality(self, source, target, te_history):
        te = pyte.transfer_entropy(source, target, te_history, local=True)
        normalizer = pyinform.entropyrate.entropy_rate(
            target, k=te_history, local=False
        )
        if np.isclose(normalizer, 0.0):
            return np.zeros_like(te)
        else:
            return te / normalizer
