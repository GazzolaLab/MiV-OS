__doc__ = """Sample signal readout from optogenetic neuron."""
__all__ = ["load_data"]

import gzip
import os

import numpy as np

from miv.datasets.utils import get_file
from miv.io.openephys import DataManager


def load_data():  # pragma: no cover
    """
    Loads the sample optogenetic experiment data. `Direct Download <https://uofi.box.com/shared/static/9llg11ods9iejdt2omjwjosbsxb5ui10.zip>`_

    Total size: 600.5 MB (compressed)

    File hash: 5deadc1b2a20501b5f6ee8828fa9c85df0b7890bd6ac4eaa8dca768d3b8b5f83

    Notes
    -----
    All experiment are 1 minute long, 30k Hz recording of optogenetic
    neuron cells over 64 channels MEA. Dataset includes 1 spontaneous
    recording and 4 stimulated recording.

    Spontaneous recording is the recording over 1 minute period without
    external stimulation. The purpose was to measure the baseline mean-
    firing rate.

    Stimulation was done by LED light. Over 1 minute (60 seconds) period,
    6 stimulation was done with 10 seconds of intervals. For each stimulation,
    LED light was shined over 1 seconds, followed by remaining 9 seconds
    of rest (without light).

    Containing experiments:

    * experiment0: spontaneous recording
    * experiment1-4: stimulated recordings

    Returns
    -------
    dataset: miv.io.DataManager

    Examples
    --------
        >>> from miv import datasets
        >>> experiments: miv.io.DataManager = datasets.optogenetic.load_data()
        >>> experiments.tree()
        2022-03-10_16-19-09
            0: <miv.io.data.Data object at 0x7fbc30b9a9a0>
               └── Record Node 104/experiment1/recording1
            1: <miv.io.data.Data object at 0x7fbc2ae16700>
               └── Record Node 104/experiment0/recording1
            2: <miv.io.data.Data object at 0x7fbc2ac9e7c0>
               └── Record Node 104/experiment2/recording2
            3: <miv.io.data.Data object at 0x7fbc2ac9edc0>
               └── Record Node 104/experiment3/recording1
            4: <miv.io.data.Data object at 0x7fbc2ac9e160>
               └── Record Node 104/experiment4/recording3

    """

    subdir = "optogenetic"
    base_url = "https://uofi.box.com/shared/static/9llg11ods9iejdt2omjwjosbsxb5ui10.zip"
    file = "2022-03-10_16-19-09.zip"
    file_hash = "5deadc1b2a20501b5f6ee8828fa9c85df0b7890bd6ac4eaa8dca768d3b8b5f83"

    path = get_file(
        file_url=base_url,
        directory=subdir,
        fname=file,
        file_hash=file_hash,
        archive_format="zip",
    )
    experiment = DataManager(path)
    experiment.tree()
    return experiment
