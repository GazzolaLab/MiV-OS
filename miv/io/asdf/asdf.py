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


class DataASDF:
    def __init__(self, data_path):
        self.data_path = data_path
        # t_start = int(sys.argv[4])
        # t_end = sys.argv[5]

    def load(self):
        # LOAD SPIKE TIME INFO
        asdf = loadmat(f"{self.data_path}", appendmat=False)["asdf_raw"]
        info = np.squeeze(asdf[len(asdf) - 1].item())
        nNeu = info[0]
        time_end = info[1]

        # CREATE RASTER
        raster = np.zeros((nNeu, time_end + 1), dtype="int32")
        # for each neuron
        for i in range(asdf.shape[0] - 2):
            # get spike times
            times = np.squeeze(asdf[i].item())
            times = times.astype(int)
            # need to modify to fix error, "tuple index out of range"- think it's if something in the asdf is empty, but not sure
            if times.size > 1:
                # set times of spikes in raster to 1
                raster[i, times] = 1
        # if t_end == "end":
        #    t_end = time_end
        # else:
        #    t_end = int(t_end)
        # raster = raster[:,t_start:t_end]
        return raster
