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

from miv.core.datatype import Spikestamps
from miv.core.pipeline import Pipeline
from miv.signal.filter import ButterBandpass
from miv.signal.spike import ThresholdCutoff


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
    signal_filter = ButterBandpass(400, 1500, order=4)
    spike_detection = ThresholdCutoff(cutoff=5.0, progress_bar=True)
    signal_filter >> spike_detection

    # Select tools
    if tools == "OpenEphys":
        # If OpenEphys is selected, process all recordings in the set.
        from miv.io.openephys import DataManager

        for p in path:
            for data in DataManager(p):
                data >> signal_filter
                pipeline = Pipeline(spike_detection)
                pipeline.run(save_path=data.analysis_path, no_cache=True)
                spike_detection.plot(save_path=data.analysis_path)
                logging.info(f"Pre-processing {p}-{data.data_path} done.")
                data.clear_connections()
    elif tools == "Intan":
        from miv.io.intan import DataIntan

        for p in path:
            data = DataIntan(p)
            pipeline = Pipeline(spike_detection)
            pipeline.run(save_path=data.analysis_path, no_cache=True)
            spike_detection.plot(save_path=data.analysis_path)
            logging.info(f"Pre-processing {p}-{data.data_path} done.")
            data.clear_connections()


if __name__ == "__main__":
    main()
