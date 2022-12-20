# A script to convert Open Ephys data to MiV HDF5 format.

import logging
import os
import sys

import click
import numpy as np

from miv.io import Data, DataManager
from miv.io import file as miv_file


def config_logging(verbose):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARN)


script_name = os.path.basename(__file__)


def seq_find(f, seq):
    """
    Determines if an element satisfying predicate is present in the
    given sequence, returns index of element or None.

    :param f: predicate function
    :param seq: sequence
    :return: index of element satisfying predicate, or None

    """
    i = 0
    for x in seq:
        if f(x):
            return i
        else:
            i = i + 1
    return None


@click.command()
@click.option("--folder-path", "-p", required=True, type=click.Path())
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(folder_path: str, verbose: bool):
    """Convert Open Ephys data to MiV HDF5 format"""

    config_logging(verbose)
    logger = logging.getLogger("miv_file")

    # Load dataset from OpenEphys recording
    data_manager = DataManager(folder_path)
    miv_data = miv_file.initialize()
    cont = miv_file.create_container(miv_data)

    # Get signal and rate(hz)
    #   signal        : np.array, shape(N, N_channels)
    #   timestamps    : np.array
    #   sampling_rate : float
    for recording in data_manager:
        data_path = recording.data_path
        with recording.load() as (signal, timestamps, sampling_rate):
            group_id = miv_file.create_group(miv_data, data_path, counter="nrecording")
            miv_file.create_dataset(
                miv_data,
                ["signal", "timestamps", "sampling_rate"],
                group=data_path,
                dtype=float,
            )
            cont[f"{group_id}/signal"] = signal
            cont[f"{group_id}/timestamps"] = timestamps
            cont[f"{group_id}/sampling_rate"] = [sampling_rate]
            miv_file.pack(miv_data, cont, logger=logger)

    miv_file.write(
        f"{folder_path}/MiV_data.h5",
        miv_data,
        comp_type="gzip",
        comp_opts=9,
        logger=logger,
    )


if __name__ == "__main__":
    main(
        args=sys.argv[
            (seq_find(lambda x: os.path.basename(x) == script_name, sys.argv) + 1) :
        ]
    )
