__doc__ = """OpenEphys sample data."""
__all__ = ["load_data"]

import gzip
import os

import numpy as np

from miv.datasets.utils import get_file
from miv.io.openephys import DataManager


def load_data(progbar_disable: bool = False):  # pragma: no cover
    """
    Loads the sample recorded data from OpenEphys aquisition system. `Direct Download <https://uofi.box.com/shared/static/daagmwebzfdl1jjgfhjzzhof2if44m04.zip>`_

    Total size: 21.2 kB (compressed)

    File hash: ab1b282a2560e75263188dc2bab56b019bb83ea12aa63692c9560c6e8281de29

    Examples
    --------
        >>> from miv.datasets.ttl_events import load_data
        >>> experiments: miv.io.DataManager = load_data()
        datasets/ttl_recording/sample_event_recording
            0: <miv.io.data.Data object at 0x7fec71c99fa0>
               └── Record Node 101/experiment1/recording1


    Notes
    -----
    All experiment are 1 minute long, 30k Hz recording of optogenetic
    neuron cells over 64 channels MEA. Dataset includes 1 spontaneous
    recording.

    Spontaneous recording is the recording over 10 seconds period.

    Containing experiments:

    * experiment0: 10 seconds spontaneous recording

    Returns
    -------
    dataset: miv.io.DataManager

    Examples
    --------
        >>> from miv import datasets
        >>> experiments: miv.io.DataManager = datasets.openephys_sample.load_data()

    """

    subdir = "OpenEphys"
    base_url = "https://uofi.box.com/shared/static/daagmwebzfdl1jjgfhjzzhof2if44m04.zip"
    file = "OpenEphys_Sample_64MEA.zip"
    file_hash = "ab1b282a2560e75263188dc2bab56b019bb83ea12aa63692c9560c6e8281de29"

    path = get_file(
        file_url=base_url,
        directory=subdir,
        fname=file,
        file_hash=file_hash,
        archive_format="zip",
        progbar_disable=progbar_disable,
    )
    experiment = DataManager(path)
    experiment.tree()
    return experiment
