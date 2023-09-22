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
from miv.core.operator.policy import StrictMPIRunner
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
            binned = spikestamps.binning(self.bin_size, return_count=True)
            probe_times = binned.timestamps[:: self.skip_interval]
        else:
            probe_times = None
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

        # Get spikestamps view
        probe_times = np.array_split(probe_times, size)[rank]
        start_time = probe_times[0]
        end_time = probe_times[-1]
        spiketrains_bins = spikestamps.get_view(start_time, end_time).binning(
            self.bin_size, return_count=True
        )
        logging.info(
            f"{rank=} | rendering from {start_time=:.03f} to {end_time=:.03f}: {probe_times.shape[0]} frames."
        )

        # Find firing rate
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
        with writer.saving(fig, video_name, dpi=200):
            for timestep, time in tqdm(
                enumerate(probe_times),
                desc=f"Rendering {rank}/{size}",
                position=rank,
                total=probe_times.shape[0],
                disable=not self.progress_bar,
            ):
                data = xs[:, timestep]
                X, Y, Z = mea.map_data(data)

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


@dataclass
class NeuralActivity512(OperatorMixin):
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
            binned = spikestamps.binning(self.bin_size, return_count=True)
            probe_times = binned.timestamps[:: self.skip_interval]
        else:
            probe_times = None
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

        # Get spikestamps view
        probe_times = np.array_split(probe_times, size)[rank]
        start_time = probe_times[0]
        end_time = probe_times[-1]
        spiketrains_bins = spikestamps.get_view(start_time, end_time).binning(
            self.bin_size, return_count=True
        )
        logging.info(
            f"{rank=} | rendering from {start_time=:.03f} to {end_time=:.03f}: {probe_times.shape[0]} frames."
        )

        # Find firing rate
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

        # DEBUG
        electrode_map_512 = (
            np.array(
                [
                    [
                        48,
                        211,
                        214,
                        217,
                        220,
                        223,
                        226,
                        229,
                        232,
                        235,
                        238,
                        16,
                        245,
                        248,
                        251,
                        254,
                        193,
                        196,
                        197,
                        200,
                        203,
                        206,
                        113,
                        179,
                        182,
                        185,
                        188,
                        191,
                        130,
                        133,
                        136,
                        139,
                        142,
                        81,
                        147,
                        150,
                        153,
                        156,
                        159,
                        162,
                        165,
                        168,
                        171,
                        174,
                        467,
                        470,
                        473,
                        476,
                        479,
                        482,
                        485,
                        488,
                        491,
                        494,
                        304,
                        499,
                        502,
                        505,
                        508,
                        511,
                        450,
                        453,
                        456,
                        459,
                        462,
                        272,
                        435,
                        438,
                        441,
                        444,
                        445,
                        448,
                        387,
                        390,
                        393,
                        396,
                        369,
                        403,
                        406,
                        409,
                        412,
                        415,
                        418,
                        421,
                        424,
                        427,
                        430,
                        337,
                    ],
                    [
                        209,
                        45,
                        42,
                        39,
                        36,
                        33,
                        30,
                        27,
                        24,
                        21,
                        18,
                        241,
                        13,
                        10,
                        7,
                        4,
                        1,
                        62,
                        59,
                        56,
                        53,
                        50,
                        177,
                        116,
                        119,
                        122,
                        125,
                        128,
                        67,
                        70,
                        73,
                        76,
                        79,
                        145,
                        84,
                        87,
                        90,
                        93,
                        96,
                        99,
                        102,
                        105,
                        108,
                        111,
                        274,
                        277,
                        280,
                        283,
                        286,
                        289,
                        292,
                        295,
                        298,
                        301,
                        496,
                        306,
                        309,
                        312,
                        315,
                        318,
                        257,
                        260,
                        263,
                        266,
                        269,
                        464,
                        335,
                        332,
                        329,
                        326,
                        323,
                        384,
                        381,
                        378,
                        375,
                        372,
                        400,
                        367,
                        364,
                        361,
                        358,
                        355,
                        352,
                        349,
                        346,
                        343,
                        340,
                        432,
                    ],
                    [
                        47,
                        212,
                        215,
                        218,
                        221,
                        224,
                        227,
                        230,
                        233,
                        236,
                        239,
                        15,
                        244,
                        247,
                        250,
                        253,
                        256,
                        195,
                        198,
                        201,
                        204,
                        207,
                        114,
                        180,
                        183,
                        186,
                        189,
                        192,
                        131,
                        134,
                        137,
                        140,
                        143,
                        82,
                        148,
                        151,
                        154,
                        157,
                        160,
                        163,
                        166,
                        169,
                        172,
                        175,
                        466,
                        469,
                        472,
                        475,
                        478,
                        481,
                        484,
                        487,
                        490,
                        493,
                        303,
                        498,
                        501,
                        504,
                        507,
                        510,
                        449,
                        452,
                        455,
                        458,
                        461,
                        271,
                        434,
                        437,
                        440,
                        443,
                        446,
                        385,
                        388,
                        391,
                        394,
                        397,
                        370,
                        402,
                        405,
                        408,
                        411,
                        414,
                        417,
                        420,
                        423,
                        426,
                        429,
                        338,
                    ],
                    [
                        210,
                        44,
                        41,
                        38,
                        35,
                        32,
                        29,
                        26,
                        23,
                        20,
                        17,
                        242,
                        12,
                        9,
                        6,
                        3,
                        64,
                        61,
                        58,
                        55,
                        52,
                        49,
                        178,
                        117,
                        120,
                        123,
                        126,
                        65,
                        68,
                        71,
                        74,
                        77,
                        80,
                        146,
                        85,
                        88,
                        91,
                        94,
                        97,
                        100,
                        103,
                        106,
                        109,
                        112,
                        273,
                        276,
                        279,
                        282,
                        285,
                        288,
                        291,
                        294,
                        297,
                        300,
                        495,
                        305,
                        308,
                        311,
                        314,
                        317,
                        320,
                        259,
                        262,
                        265,
                        268,
                        463,
                        336,
                        333,
                        330,
                        327,
                        324,
                        321,
                        382,
                        379,
                        376,
                        373,
                        399,
                        368,
                        365,
                        362,
                        359,
                        356,
                        353,
                        350,
                        347,
                        344,
                        341,
                        431,
                    ],
                    [
                        46,
                        213,
                        216,
                        219,
                        222,
                        225,
                        228,
                        231,
                        234,
                        237,
                        240,
                        14,
                        243,
                        246,
                        249,
                        252,
                        255,
                        194,
                        199,
                        202,
                        205,
                        208,
                        115,
                        181,
                        184,
                        187,
                        190,
                        129,
                        132,
                        135,
                        138,
                        141,
                        144,
                        83,
                        149,
                        152,
                        155,
                        158,
                        161,
                        164,
                        167,
                        170,
                        173,
                        176,
                        465,
                        468,
                        471,
                        474,
                        477,
                        480,
                        483,
                        486,
                        489,
                        492,
                        302,
                        497,
                        500,
                        503,
                        506,
                        509,
                        512,
                        451,
                        454,
                        457,
                        460,
                        270,
                        433,
                        436,
                        439,
                        442,
                        447,
                        386,
                        389,
                        392,
                        395,
                        398,
                        371,
                        401,
                        404,
                        407,
                        410,
                        413,
                        416,
                        419,
                        422,
                        425,
                        428,
                        339,
                    ],
                    [
                        0,
                        43,
                        40,
                        37,
                        34,
                        31,
                        28,
                        25,
                        22,
                        19,
                        0,
                        0,
                        11,
                        8,
                        5,
                        2,
                        63,
                        60,
                        57,
                        54,
                        51,
                        0,
                        0,
                        118,
                        121,
                        124,
                        127,
                        66,
                        69,
                        72,
                        75,
                        78,
                        0,
                        0,
                        86,
                        89,
                        92,
                        95,
                        98,
                        101,
                        104,
                        107,
                        110,
                        0,
                        0,
                        275,
                        278,
                        281,
                        284,
                        287,
                        290,
                        293,
                        296,
                        299,
                        0,
                        0,
                        307,
                        310,
                        313,
                        316,
                        319,
                        258,
                        261,
                        264,
                        267,
                        0,
                        0,
                        334,
                        331,
                        328,
                        325,
                        322,
                        383,
                        380,
                        377,
                        374,
                        0,
                        0,
                        366,
                        363,
                        360,
                        357,
                        354,
                        351,
                        348,
                        345,
                        342,
                        0,
                    ],
                ],
                dtype=np.int_,
            )
            - 1
        )
        for a, b in zip(range(288, 319 + 1), range(287, 256 + 1)):
            ma = electrode_map_512 == a
            mb = electrode_map_512 == b
            electrode_map_512[ma] = b
            electrode_map_512[mb] = a
        for a, b in zip(range(96, 127 + 1), range(95, 64 + 1)):
            ma = electrode_map_512 == a
            mb = electrode_map_512 == b
            electrode_map_512[ma] = b
            electrode_map_512[mb] = a
        nr, nt = electrode_map_512.shape
        it = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, nt + 1)[::-1]  # pcb flipped
        it = (it[1:] + it[:-1]) / 2.0
        ir = np.linspace(4950, 3300, nr)
        tv, rv = np.meshgrid(it, ir)
        xv = rv * np.cos(tv) + 1650
        yv = rv * np.sin(tv) - 1650

        def map_electrode_data(vector, missing_value=0.0):
            grid = electrode_map_512
            value_grid = np.full_like(grid, missing_value, dtype=np.float_)
            for idx, value in enumerate(vector):
                if idx not in grid:
                    continue
                value_grid[grid == idx] = value
            return xv, yv, value_grid

        # DEBUG
        if self.runner.is_root():
            plt.figure()
            plt.plot(xv.ravel(), yv.ravel(), "k.", ms=1)
            plt.axvline(0, color="red", linestyle=":")
            plt.axhline(0, color="red", linestyle=":")
            plt.savefig(os.path.join(self.analysis_path, "grid_electrodes.png"))
            plt.close("all")

            # Test
            index = np.arange(xs.shape[0])

            X, Y, Z = mea.map_data(index)
            fig, ax = plt.subplots(1, 1, figsize=(16, 16))
            ax.scatter(X.ravel(), Y.ravel())
            for i, txt in enumerate(Z.ravel()):
                ax.annotate(str(int(txt)), (X.ravel()[i], Y.ravel()[i]))
            X, Y, Z = map_electrode_data(index)
            ax.scatter(X.ravel(), Y.ravel())
            for i, txt in enumerate(Z.ravel()):
                ax.annotate(str(int(txt)), (X.ravel()[i], Y.ravel()[i]))
            plt.axvline(0, color="red", linestyle=":")
            plt.axhline(0, color="red", linestyle=":")
            ax.set_aspect("equal")
            plt.savefig(os.path.join(self.analysis_path, "grid.png"))
            plt.close("all")

            X, Y, Z = mea.map_data(index)
            eX, eY, eZ = map_electrode_data(index)
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.plot(mea.coordinates[:, 0], mea.coordinates[:, 1], "k.", ms=1)
            pcm = ax.pcolormesh(
                X,
                Y,
                Z,
                cmap="twilight",
                vmin=index.min(),
                vmax=index.max(),
                shading="gouraud",
            )
            ax.scatter(
                eX.ravel(),
                eY.ravel(),
                c=eZ.ravel(),
                cmap="twilight",
                vmin=index.min(),
                vmax=index.max(),
            )
            cbar = fig.colorbar(pcm, ax=ax)
            cbar.ax.set_ylabel(
                f"activity per {self.firing_rate_interval:.03f} sec", rotation=270
            )

            ax.set_aspect("equal")
            ax.set_xlabel("channels x-axis")
            ax.set_ylabel("channels y-axis")
            ax.set_title("Spatial Neural Activity (--)")
            plt.savefig(os.path.join(self.analysis_path, "grid_test.png"))
            plt.close("all")
        # DEBUG

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
        with writer.saving(fig, video_name, dpi=200):
            for timestep, time in tqdm(
                enumerate(probe_times),
                desc=f"Rendering {rank}/{size}",
                position=rank,
                total=probe_times.shape[0],
                disable=not self.progress_bar,
            ):
                data = xs[:, timestep]
                X, Y, Z = mea.map_data(data)
                eX, eY, eZ = map_electrode_data(data)

                fig.clf()
                ax = fig.add_subplot(111)
                # X, Y, Z = interp_2d(Z)
                ax.plot(mea.coordinates[:, 0], mea.coordinates[:, 1], "k.", ms=1)
                pcm = ax.pcolormesh(
                    X, Y, Z, cmap="Oranges", vmin=xmin, vmax=xmax, shading="gouraud"
                )
                ax.scatter(
                    eX.ravel(),
                    eY.ravel(),
                    c=eZ.ravel(),
                    cmap="Oranges",
                    vmin=xmin,
                    vmax=xmax,
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
