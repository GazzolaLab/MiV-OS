__doc__ = """

Module (MiV-Simulator)
######################

.. autoclass:: Data
   :members:

"""
__all__ = ["Data"]

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import logging
import os
import pickle
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from miv.core.datatype import Signal, Spikestamps
from miv.core.operator import DataLoaderMixin


class Data(DataLoaderMixin):
    """Single result from miv-simulator.

    Parameters
    ----------
    data_path : str

    """

    def __init__(self, data_path: str, *args, **kwargs):
        self.data_path = data_path
        super().__init__(*args, **kwargs)

        self._lfp_key = "Local Field Potential"  # TODO: refactor
        self._load_every = 60  # sec. Parse every 60 sec.

    def load(self):
        yield from self.load_lfp_recordings()

    def load_lfp_recordings(self, indices: Optional[List[int]] = None):
        infile = h5py.File(self.data_path)

        if indices is not None:
            keys = [
                key
                for key in infile.keys()
                if self._lfp_key in key and int(key.split(" ")[-1]) in indices
            ]
        else:
            keys = [key for key in infile.keys() if self._lfp_key in key]

        # Check if t matches
        t0 = None
        for _, namespace_id in enumerate(keys):
            if t0 is None:
                t0 = np.asarray(infile[namespace_id]["t"])
                continue
            t = np.asarray(infile[namespace_id]["t"])
            if not np.allclose(t0, t):
                raise ValueError(
                    "Recorded time for electrodes does not match."
                    "Check if the sampling rates for each electrode are the same."
                )

        sampling_rate = 1000.0 / np.median(
            np.diff(t0)
        )  # FIXME: Try to infer from environment configuration instead
        length = int(self._load_every * sampling_rate)
        findex = len(t0)
        sindex, eindex = 0, min(length, findex)
        while eindex <= findex:
            signals = []
            for _, namespace_id in enumerate(keys):
                v = np.asarray(infile[namespace_id]["v"])
                signals.append(v[sindex:eindex])
            yield Signal(
                data=np.asarray(signals).T,
                timestamps=t0[sindex:eindex],
                rate=sampling_rate,
            )
            sindex += length
            eindex = min(eindex + length, len(t0))

    def check_path_validity(self):
        """
        Check if necessary files exist in the directory.

        Returns
        -------
        bool
            Return true if all necessary files exist in the directory.
        """
        return os.path.exists(self.data_path)
