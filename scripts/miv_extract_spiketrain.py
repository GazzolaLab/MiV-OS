import os
import sys
from glob import glob

import click
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import muptiprocessing as mp
import numpy as np
import scipy as sp
from tqdm import tqdm

from miv.core import Spikestamps
from miv.signal.filter import ButterBandpass, FilterCollection, MedianFilter
from miv.signal.spike import ThresholdCutoff


def preprocessing(data):
    signal, timestamps, sampling_rate = data
    signal_filter = ButterBandpass(400, 1500, order=4)
    spike_detection = ThresholdCutoff(cutoff=5)
    filtered_signal = signal_filter(signal, sampling_rate)
    spikestamp = spike_detection(
        filtered_signal,
        timestamps,
        sampling_rate,
        return_neotype=False,
        progress_bar=False,
    )
    return spikestamp


@click.command()
@click.option("--path", "-p", help="Path to the recorded directory.", multiple=True)
@click.option(
    "--tools",
    "-t",
    type=click.Choice(["OpenEphys", "Intan"]),
    default="OpenEphys",
    help="Select the recording format",
)
@click.option(
    "--nproc",
    "-N",
    default=None,
    help="Number of processing units. If not provided, use maximum.",
)
@click.option(
    "--num_fragments",
    "-n",
    default=1,
    help="Number of fragments for data processing. Recommend to split data into 1 minute recording segments. For Intan, this number is pre-determined by the number of saved files.",
)
@click.option(
    "--use-mpi",
    type=bool,
    default=False,
    help="Set True if mpi is ready. Else, it will use multiprocessing. (mpi4py must be installed)",
)
@click.option("--chunksize", default=1, help="Number of chunks for multiprocessing")
def main(path, tools, nproc, num_fragments, use_mpi, chunksize):
    # Select tools
    if tools == "OpenEphys":
        # If OpenEphys is selected, process all recordings in the set.
        from miv.io import DataManager

        raise NotImplementedError
    elif tools == "Intan":
        from miv.io.intan import DataIntan

        for p in path:
            total_spikestamps = Spikestamps([])
            data = DataIntan(p)
            with mp.Pool(nproc) as pool:
                for spikestamp in pool.imap(
                    preprocessing, data.load(), chunksize=chunksize
                ):
                    total_spikestamps.extend(spikestamp)
            data.save_data(total_spikestamps, "spiketrain")


if __name__ == "__main__":
    main()
