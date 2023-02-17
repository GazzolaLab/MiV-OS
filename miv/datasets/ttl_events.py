__doc__ = """Sample TTL-input signal readout."""
__all__ = ["load_data"]

import gzip
import os

import numpy as np

from miv.datasets.utils import get_file
from miv.io.openephys import DataManager


def load_data():  # pragma: no cover
    """
    Loads the sample TTL data readout. `Direct Download <https://uofi.box.com/shared/static/9llg11ods9iejdt2omjwjosbsxb5ui10.zip>`_

    Total size: 17.4 kB (compressed)

    File hash: a4314442fd9eba4377934c5766971ba1e04f079b7e615b8fb033992323afeb3f

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
    recording and 4 stimulated recording.

    Spontaneous recording is the recording over 1 minute period without
    external stimulation. The purpose was to measure the baseline mean-
    firing rate.

    Stimulation was done by LED light. Over 1 minute (60 seconds) period,
    6 stimulation was done with 10 seconds of intervals. For each stimulation,
    LED light was shined over 1 seconds, followed by remaining 9 seconds
    of rest (without light).

    Containing experiments:

    * experiment0: TTL recording

    Returns
    -------
    dataset: miv.io.DataManager

    Examples
    --------
        >>> from miv import datasets
        >>> experiments: miv.io.DataManager = datasets.ttl_events.load_data()

    """

    subdir = "ttl_recording"
    base_url = "https://uofi.box.com/shared/static/w3cylplece450up6t6h53vuq93t98q2k.zip"
    file = "sample_event_recording.zip"
    file_hash = "a4314442fd9eba4377934c5766971ba1e04f079b7e615b8fb033992323afeb3f"

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
