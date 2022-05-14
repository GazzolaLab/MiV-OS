__doc__ = """

.. Note::
    We expect the data structure to follow the default format
    exported from OpenEphys system:
    `format <https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/491632/Data+format>`_.

.. Note::
    For simple experiments, you may prefer to use :ref:`api/io:Raw Data Loader`.
    However, we generally recommend to use ``Data`` or ``DataManager`` for
    handling data, especially when the size of the raw data is large.

Module
######

.. currentmodule:: miv.io.data

.. autoclass:: Data
   :members:

.. autoclass:: DataManager
   :members:

"""
__all__ = ["Data", "DataManager"]

from typing import Any, Optional, Iterable, Callable, List

from collections.abc import MutableSequence
import logging

import os
from glob import glob
import numpy as np
from contextlib import contextmanager

from miv.io.binary import load_continuous_data, load_recording
from miv.signal.filter import FilterProtocol
from miv.typing import SignalType


class Data:
    """Single data unit handler.

    Each data unit that contains single recording. This class provides useful tools,
    such as masking channel, export data, interface with other packages, etc.
    If you have multiple recordings you would like to handle at the same time, use
    `DataManager` instead.

    By default recording setup, the following directory structure is expected in ``data_path``::

        recording1                              # <- recording data_path
        ├── continuous
        │   └── Rhythm_FPGA-100.0
        │       ├── continuous.dat
        │       ├── synchronized_timestamps.npy
        │       └── timestamps.npy
        ├── events
        │   ├── Message_Center-904.0
        │   │   └── TEXT_group_1
        │   │       ├── channels.npy
        │   │       ├── text.npy
        │   │       └── timestamps.npy
        │   └── Rhythm_FPGA-100.0
        │       └── TTL_1
        │           ├── channel_states.npy
        │           ├── channels.npy
        │           ├── full_words.npy
        │           └── timestamps.npy
        ├── sync_messages.txt
        ├── structure.oebin
        └── analysis                            # <- post-processing result
            ├── spike_data.npz
            ├── plot
            ├── spike
            └── mea_overlay


        Parameters
        ----------
        data_path : str
    """

    def __init__(
        self,
        data_path: str,
    ):
        self.data_path = data_path
        self.channel_mask = []

    @contextmanager
    def load(self):
        """
        Context manager for loading data instantly.

        Examples
        --------
            >>> data = Data(data_path)
            >>> with data.load() as (timestamps, raw_signal):
            ...     ...

        Returns
        -------
        signal : SignalType, neo.core.AnalogSignal
        timestamps : TimestampsType, numpy array
        sampling_rate : float

        """
        # TODO: Not sure this is safe implementation
        try:
            signal, timestamps, sampling_rate = load_recording(
                self.data_path, self.masking_channel_list
            )
            yield signal, timestamps, sampling_rate
        except FileNotFoundError as e:
            logging.error(
                f"The file could not be loaded because the file {self.data_path} does not exist."
            )
            logging.error(e.strerror)
        except ValueError as e:
            logging.error(
                "The data size does not match the number of channel. Check if oebin or continuous.dat file is corrupted."
            )
            logging.error(e.strerror)
        finally:
            del timestamps
            del signal

    def set_channel_mask(self, channel_id: List[int]):
        """

        Set the channel masking.

        Notes
        -----
        If the index exceed the number of channels, it will be ignored.

        Parameters
        ----------
        channel_id : List[int]
            List of channel id that will be ignored.
        """
        self.masking_channel_list = channel_id

    def save(self, tag: str, format: str):
        assert tag == "continuous", "You cannot alter raw data, change the data tag"
        # save_path = os.path.join(self.data_path, tag)

        if format == "dat":
            ...
        elif format == "npz":
            ...
        elif format == "neo":
            ...
        else:
            raise NotImplementedError(
                "Format type " + format + " is not implemented.\n",
                "Please choose  one of the supported formats: dat, npz, neo",
            )


class DataManager(MutableSequence):
    def __init__(
        self,
        data_folder_path: str,
        channels: int,
        sampling_rate: float,
        timestamps_npy: Optional[str] = "",
        device="",
    ):
        self.data_folder_path = data_folder_path

        # From the path get data paths and create data objects
        self.load_data_sets(channels, sampling_rate, timestamps_npy)

    def load_data_sets(self, channels, sampling_rate, timestamps_npy):
        """
        Create data objects from the data three.

        Parameters
        ----------
        path

        Returns
        -------

        """
        # From the path get the data path list
        self.data_path_list = self._get_data_path_from_tree()

        # Create an object for each continues.dat and store them in data list to manipulate later.
        self.data_list = []
        for data_path in self.data_path_list:
            self.data_list.append(
                Data(data_path, channels, sampling_rate, timestamps_npy)
            )

    def _get_data_path_from_tree(self):
        """
        This function gets the data for each continues.dat file inside the data folder.
        Returns
        -------
        data_path_list : list
        """
        # TODO: implement algorithm to get paths of all continues.dat files.
        # Use self.data_folder_path
        raise NotImplementedError("Loading data tree not implemented yet")
        data_path_list = []
        return data_path_list

    def save(self, tag: str, format: str):
        for data in self.data_list:
            data.save(tag, format)

    def apply_filter(self, filter: FilterProtocol):
        for data in self.data_list:
            data.load()
            data = filter(data, sampling_rate=0)
            data.save(tag="filter", format="npz")
            data.unload()

    # def apply_spike_detection(self, method: DetectionProtocol):
    #     raise NotImplementedError("Wait until we make it")

    # MutableSequence abstract methods
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __delitem__(self, idx):
        del self.data_list[idx]

    def __setitem__(self, idx, system):
        self.data_list[idx] = system

    def __call__(self, *args, **kwargs):
        pass


def get_experiments_recordings(data_paths: str) -> Iterable[str]:
    # fmt: off
    list_of_experiments_to_process = []
    for path in data_paths:
        path_list = [path for path in glob.glob(os.path.join(path, "*", "*", "*")) if "Record Node" in path and "recording" in path and os.path.isdir(path)]
        list_of_experiments_to_process.extend(path_list)
    # fmt: on
    return list_of_experiments_to_process


def get_analysis_paths(data_paths: str, output_folder_name: str) -> Iterable[str]:
    # fmt: off
    list_of_analysis_paths = []
    for path in data_paths:
        path_list = [path for path in glob.glob(os.path.join(path, "*", "*", "*", "*")) if ("Record Node" in path) and ("recording" in path) and (output_folder_name in path) and os.path.isdir(path)]
        list_of_analysis_paths.extend(path_list)
    # fmt: on
    return list_of_analysis_paths
