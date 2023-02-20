import logging
import multiprocessing as mp
import os
import sys
from glob import glob

import click
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from tqdm import tqdm

from miv.core.operator import Operator
from miv.core.datatype import Spikestamps
from miv.signal.filter import ButterBandpass
from miv.signal.spike import ThresholdCutoff


def preprocessing(data):
    signal, timestamps, sampling_rate = data

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
    signal_filter:Operator = ButterBandpass(400, 1500, order=4)
    spike_detection:Operator = ThresholdCutoff(cutoff=5)
    preprocessing:Operator = signal_filter >> spike_detection
    # Select tools
    if tools == "OpenEphys":
        # If OpenEphys is selected, process all recordings in the set.
        from miv.io.openephys import DataManager

        raise NotImplementedError
    elif tools == "Intan":
        from miv.io.intan import DataIntan

        for p in path:
            data:Operator = DataIntan(p)
            pipeline = data >> preprocessing
            pipeline.run(num_processors=nproc)
            logging.info(f"Pre-processing {p} done.")


if __name__ == "__main__":
    main()
