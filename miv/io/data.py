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

----------------------

.. autoclass:: DataManager
   :members:

"""
__all__ = ["Data", "DataManager"]

from typing import Any, Callable, Dict, Iterable, List, Optional, Set

import logging
import os
from collections.abc import MutableSequence
from contextlib import contextmanager
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from miv.io.binary import load_continuous_data, load_recording
from miv.signal.filter.protocol import FilterProtocol
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
        self.data_path: str = data_path
        self.analysis_path: str = os.path.join(data_path, "analysis")
        self.masking_channel_set: Set[int] = set()

        os.makedirs(self.analysis_path, exist_ok=True)

    def save_figure(
        self,
        figure: plt.Figure,
        group: str,
        filename: str,
        savefig_kwargs: Optional[Dict[Any, Any]] = None,
    ):
        """Save figure in analysis sub-directory

        Parameters
        ----------
        figure : plt.Figure
        group : str
        filename : str
        """
        if savefig_kwargs is None:
            savefig_kwargs = {}

        dirpath = os.path.join(self.analysis_path, group)
        os.makedirs(dirpath, exist_ok=True)

        filepath = os.path.join(dirpath, filename)
        plt.figure(figure)
        plt.savefig(filepath, **savefig_kwargs)

    @contextmanager
    def load(self):
        """
        Context manager for loading data instantly.

        Examples
        --------
            >>> data = Data(data_path)
            >>> with data.load() as (signal, timestamps, sampling_rate):
            ...     ...

        Returns
        -------
        signal : SignalType, neo.core.AnalogSignal
        timestamps : TimestampsType, numpy array
        sampling_rate : float

        Raises
        ------
        FileNotFoundError
            If some key files are missing.

        """
        # TODO: Not sure this is safe implementation
        if not self.check_path_validity():
            raise FileNotFoundError("Data directory does not have all necessary files.")
        try:
            signal, timestamps, sampling_rate = load_recording(
                self.data_path, self.masking_channel_set
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
            logging.error(e)
        finally:
            del timestamps
            del signal

    def set_channel_mask(self, channel_id: Iterable[int]):
        """
        Set the channel masking.

        Parameters
        ----------
        channel_id : Iterable[int], list
            List of channel id that will be ignored.

        Notes
        -----
        If the index exceed the number of channels, it will be ignored.

        Examples
        --------
        >>> data = Data(data_path)
        >>> data.set_channel_mask(range(12,23))

        """
        self.masking_channel_set.update(channel_id)

    def save(self, tag: str, format: str):  # TODO
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

    def check_path_validity(self):
        """
        Check if necessary files exist in the directory.

        - Check `continious.dat` exists. (only one)
        - Check `structure.oebin` exists.

        Returns
        -------
        bool
            Return true if all necessary files exist in the directory.

        """

        continuous_dat_paths = glob(
            os.path.join(self.data_path, "**", "continuous.dat"), recursive=True
        )
        if len(continuous_dat_paths) != 1:
            logging.warning(
                f"One and only one continuous.dat file can exist in the data path. Found: {continuous_dat_paths}"
            )
            return False
        if not os.path.exists(os.path.join(self.data_path, "structure.oebin")):
            logging.warning("Missing structure.oebin in the data path.")
            return False
        return True


class DataManager(MutableSequence):
    """
    Data collection manager.

    By default recording setup, the directory is named after the date and time
    of the recording. The structure of ``data_collection_path`` typically look
    like below::

        2022-03-10_16-19-09         <- data_collection_path
        └── Record Node 104
            └── experiment1
                └── recording1      <- data_path (Data module)
            ├── experiment2
            ├── experiment3
            ├── experiment4
            ├── spontaneous
            ├── settings.xml
            ├── settings_2.xml
            └── settings_3.xml

        Parameters
        ----------
        data_collection_path : str
            Path for data collection.

    """

    def __init__(self, data_collection_path: str):
        self.data_collection_path = data_collection_path
        self.data_list: Iterable[Data] = []

        # From the path get data paths and create data objects
        self._load_data_paths()

    @property
    def data_path_list(self) -> Iterable[str]:
        return [data.data_path for data in self.data_list]

    # Queries
    def query_path_name(self, query_path) -> Iterable[Data]:
        return list(filter(lambda d: query_path in d.data_path, self.data_list))

    # DataManager Representation
    def tree(self):
        """
        Pretty-print available recordings in DataManager in tree format.

        Examples
        --------
        >>> data_collection = DataManager("2022-05-15_13-51-36")
        >>> data_collection.tree()
        2022-05-15_14-51-36
            0: <miv.io.data.Data object at 0x7f8960660cd0>
               └── Record Node 103/experiment3_std2_pt_ESC/recording1
            1: <miv.io.data.Data object at 0x7f89671c8400>
               └── Record Node 103/experiment2_std1_pt_ESC/recording1
            2: <miv.io.data.Data object at 0x7f896199e7c0>
               └── Record Node 103/experiment1_cont_ESC/recording1

        """
        # TODO: Either use logging or other str stream
        if not self.data_list:
            print(
                "Data list is empty. Check if data_collection_path exists and correct"
            )
            return
        print(self.data_collection_path)
        for idx, data in enumerate(self.data_list):
            print(" " * 4 + f"{idx}: {data}")
            print(
                " " * 4
                + "   └── "
                + data.data_path[len(self.data_collection_path) + 1 :]
            )

    def _load_data_paths(self):
        """
        Create data objects from the data three.
        """
        # From the path get the data path list
        data_path_list = self._get_experiment_paths()

        # Create data object
        self.data_list = []
        invalid_count = 0
        for path in data_path_list:
            data = Data(path)
            if data.check_path_validity():
                self.data_list.append(data)
            else:
                invalid_count += 1
        logging.info(
            f"Total {len(data_path_list)} recording found. There are {invalid_count} invalid paths."
        )

    def _get_experiment_paths(self) -> Iterable[str]:
        """
        Get experiment paths.

        Returns
        -------
        data_path_list : list
        """
        # Use self.data_collection_path
        path_list = []
        for path in glob(
            os.path.join(self.data_collection_path, "*", "experiment*", "recording*")
        ):
            if (
                ("Record Node" in path)
                and ("experiment" in path)
                and os.path.isdir(path)
            ):
                path_list.append(path)
        return path_list

    def save(self, tag: str, format: str):
        raise NotImplementedError  # TODO
        for data in self.data_list:
            data.save(tag, format)

    def apply_filter(self, filter: FilterProtocol):
        raise NotImplementedError  # TODO
        for data in self.data_list:
            data.load()
            data = filter(data, sampling_rate=0)
            data.save(tag="filter", format="npz")
            data.unload()

    # MutableSequence abstract methods
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __delitem__(self, idx):
        del self.data_list[idx]

    def __setitem__(self, idx, data):
        if data.check_path_validity():
            self.data_list[idx] = data
        else:
            logging.warning("Invalid data cannot be loaded to the DataManager.")

    def insert(self, idx, data):
        if data.check_path_validity():
            self.data_list.insert(idx, data)
        else:
            logging.warning("Invalid data cannot be loaded to the DataManager.")
