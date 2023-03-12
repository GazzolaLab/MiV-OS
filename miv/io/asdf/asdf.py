__doc__ = """
ASDF file type loader

@author: tvarley
         djhavert
"""
__all__ = ["DataASDF"]

import os
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat

from miv.core.datatype import Spikestamps
from miv.core.operator import DataLoaderMixin


class DataASDF(DataLoaderMixin):
    def __init__(self, data_path, rate: float):  # pragma: no cover
        self.data_path = data_path
        self.rate = rate

    def load(self):  # pragma: no cover
        """
        Load data from ASDF file
        """
        # load spike time info
        asdf = loadmat(f"{self.data_path}", appendmat=False)["asdf_raw"]
        # info = np.squeeze(asdf[len(asdf) - 1].item())
        # nNeu = info[0]
        # time_end = info[1]

        # create spikestamps
        spikestamps = Spikestamps()
        # for each neuron
        for i in range(asdf.shape[0] - 2):
            # get spike times
            times = np.squeeze(asdf[i].item())
            times = times.astype(int) / self.rate
            # need to modify to fix error, "tuple index out of range"- think it's if something in the asdf is empty, but not sure
            spikestamps.append(times)
        return spikestamps
