__doc__ = """

Module (Intan)
##################

.. autoclass:: DataIntan
   :members:

"""
__all__ = ["DataIntan"]

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import logging
import os
import pickle
import xml.etree.ElementTree as ET
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import miv.io.intan.rhs as rhs
from miv.core.datatype import Signal
from miv.io.openephys.data import Data, DataManager
from miv.typing import SignalType


class DataManagerIntan(DataManager):
    pass


class DataIntan(Data):
    """Single data unit handler, collected from Intan.

    Each data unit that contains single recording. This class provides useful tools,
    such as masking channel, export data, interface with other packages, etc.
    If you have multiple recordings you would like to handle at the same time, use
    `DataManager` instead.

    By default recording setup, the following directory structure is expected in ``data_path``::

        data_<date>_<time>              # <- recorded data path
        ├── settings.xml
        ├── data_<date>_<time1>.rhs      # <- first rhs file
        ├── data_<date>_<time2>.rhs      # <- second rhs file
        ├── data_<date>_<time3>.rhs      # <- third rhs file
        ⋮
        └── analysis

        Parameters
        ----------
        data_path : str
    """

    def load(self):
        """
        Iterator to load data fragmentally.
        This function loads each file separately.

        Examples
        --------
            >>> data = Data(data_path)
            >>> for data.load(10) as (signal, timestamps, sampling_rate):
            ...     ...

        Returns
        -------
        signal : SignalType, neo.core.AnalogSignal
            The length of the first axis `signal.shape[0]` correspond to the length of the
            signal, while second axis `signal.shape[1]` correspond to the number of channels.
        timestamps : TimestampsType, numpy array
        sampling_rate : float

        Raises
        ------
        FileNotFoundError
            If some key files are missing.
        """

        yield from self._generator_by_channel_name("amplifier_data")

    def get_stimulation(self, progress_bar=False):
        """
        Load stimulation recorded data.
        """
        signals, timestamps = [], []
        for signal, timestamp, sampling_rate in self._generator_by_channel_name(
            "stim_data", progress_bar
        ):
            signals.append(signal)
            timestamps.append(timestamp)

        return (
            np.concatenate(signals, axis=0),
            np.concatenate(timestamps),
            sampling_rate,
        )

    def _generator_by_channel_name(self, name: str, progress_bar: bool = False):
        if not self.check_path_validity():
            raise FileNotFoundError("Data directory does not have all necessary files.")
        files = self.get_recording_files()
        files.sort()
        # Get sampling rate from setting file
        setting_path = os.path.join(self.data_path, "settings.xml")
        sampling_rate = int(ET.parse(setting_path).getroot().attrib["SampleRateHertz"])
        # Read each files
        for filename in tqdm(files, disable=not progress_bar):
            result, data_present = rhs.load_file(filename)
            assert data_present, f"Data does not present: {filename=}."
            assert not hasattr(result, name), f"No {name} in the file ({filename=})."

            # signal_group = result["amplifier_channels"]
            yield Signal(
                data=np.asarray(result[name]).T,
                timestamps=np.asarray(result["t"]),
                rate=sampling_rate,
            )

    def check_path_validity(self):
        """
        Check if necessary files exist in the directory.

        - Check `rhs` file exists.
        - Check `settings.xml` file exists.

        Returns
        -------
        bool
            Return true if all necessary files exist in the directory.
        """

        continuous_dat_paths = self.get_recording_files()
        if len(continuous_dat_paths) == 0:
            logging.warning("At least one .rhs file must exist in the data path.")
            return False
        if not os.path.exists(os.path.join(self.data_path, "settings.xml")):
            logging.warning("Missing settings.xml in the data path.")
            return False
        return True

    def get_recording_files(self):
        return glob(os.path.join(self.data_path, "*.rhs"), recursive=True)

    # Disable
    def load_ttl_event(self):
        raise AttributeError("DataIntan does not have laod_ttl_event method")
