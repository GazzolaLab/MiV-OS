from __future__ import annotations

__doc__ = """

Data Manager (OpenEphys)
########################

.. autoclass:: DataManager
   :members:

Module (OpenEphys)
##################

.. Note::
    We expect the data structure to follow the default format
    exported from OpenEphys system:
    `format <https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/491632/Data+format>`_.

.. currentmodule:: miv.io.openephys.data

.. autoclass:: Data
   :members:

"""

__all__ = ["Data", "DataManager"]


import logging
import os
import math
import re
from collections.abc import MutableSequence
from glob import glob
from typing import (
    TYPE_CHECKING,
    Any,
)
from collections.abc import Iterable, Generator

import numpy as np

from miv.core.datatype.signal import Signal
from miv.core.operator.operator import DataLoaderMixin
from miv.io.openephys.binary import load_recording, load_ttl_event
from miv.io.protocol import DataProtocol
from .binary import load_timestamps, oebin_read

if TYPE_CHECKING:
    import mpi4py


class Data(DataLoaderMixin):
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
        tag: str = "data",
    ) -> None:
        self.data_path: str = data_path
        self.tag: str = f"{tag}"
        super().__init__()
        self.masking_channel_set: set[int] = set()

    def num_fragments(self) -> int:
        # Refactor
        file_path: list[str] = glob(
            os.path.join(self.data_path, "**", "continuous.dat"), recursive=True
        )
        assert len(file_path) == 1, (
            f"There should be only one 'continuous.dat' file. (There exists {file_path})"
        )

        # load structure information dictionary
        info_file: str = os.path.join(self.data_path, "structure.oebin")
        info: dict[str, Any] = oebin_read(info_file)
        num_channels: int = info["continuous"][0]["num_channels"]
        sampling_rate: int = int(info["continuous"][0]["sample_rate"])
        # channel_info: dict[str, Any] = info["continuous"][0]["channels"]

        _old_oe_version = False

        # Read timestamps first
        dirname = os.path.dirname(file_path[0])
        timestamps_path = os.path.join(dirname, "timestamps.npy")
        timestamps = load_timestamps(timestamps_path, sampling_rate, _old_oe_version)
        total_length = timestamps.size

        # Define task
        filesize = os.path.getsize(file_path[0])
        itemsize = np.dtype("int16").itemsize
        assert filesize == itemsize * total_length * num_channels, (
            f"{filesize=} does not match the expected {itemsize*total_length*num_channels=}. Path: {file_path[0]}"
        )
        samples_per_block = sampling_rate * 60
        num_fragments = int(math.ceil(total_length / samples_per_block))
        return num_fragments

    @property
    def number_of_channels(self):
        info_file = os.path.join(self.data_path, "structure.oebin")
        info = oebin_read(info_file)
        return info["continuous"][0]["num_channels"]

    def load(
        self,
        start_at_zero: bool = False,
        progress_bar: bool = False,
        mpi_comm: mpi4py.MPI.Comm | None = None,
    ) -> Generator[Signal]:
        """
        Iterator to load data fragmentally.

        Parameters
        ----------
        start_at_zero : bool
            If set to True, time first timestamps will be shifted to zero. To achieve synchronized
            timestamps with other recordings/events, set this to False.
        progress_bar : bool
            Visible progress bar
        mpi_comm: Optional[MPI.COMM_WORLD]
            (Experimental Feature) If the load is executed in MPI environment, user can pass
            MPI.COMM_WORLD to split the load.
            For example, if num_fragments=100, rank=2, and size=10, this function will only iterate
            the fragment 20-29 out of 0-100.

        Examples
        --------
            >>> data = Data(data_path)
            >>> for data.load(num_fragments=10) as (signal, timestamps, sampling_rate):
            ...     ...

        Returns
        -------
        signal : SignalType, neo.core.AnalogSignal
            The length of the first axis `signal.shape[0]` correspond to the length of the
            signal, while second axis `signal.shape[1]` correspond to the number of channels.
        timestamps : numpy array
        sampling_rate : float

        Raises
        ------
        FileNotFoundError
            If some key files are missing.
        """
        # TODO: Not sure this is safe implementation
        if not self.check_path_validity():
            raise FileNotFoundError("Data directory does not have all necessary files.")

        for signal, timestamps, rate in load_recording(
            self.data_path,
            self.masking_channel_set,
            start_at_zero=start_at_zero,
            progress_bar=progress_bar,
            mpi_comm=mpi_comm,
        ):
            yield Signal(data=signal, timestamps=timestamps, rate=rate)

    def load_ttl_event(self) -> Signal:
        """
        Load TTL event data if data contains. Detail implementation is :func:`here <miv.io.binary.load_ttl_event>`.
        """
        states, full_words, timestamps, sampling_rate, initial_state = load_ttl_event(
            self.data_path
        )
        if timestamps.size == 0:
            self.logger.warning(
                f"TTL event was loaded but data is empty: {self.data_path}"
            )
        return Signal(data=states[:, None], timestamps=timestamps, rate=sampling_rate)

    def set_channel_mask(self, channel_id: Iterable[int]) -> None:
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

    def clear_channel_mask(self) -> None:
        """
        Clears all present channel masks.
        """
        self.masking_channel_set = set()

    def check_path_validity(self) -> bool:
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

    def __init__(self, data_collection_path: str) -> None:
        self.tag = "data_collection"
        super().__init__()
        self.data_collection_path: str = data_collection_path

        # From the path get data paths and create data objects
        self.data_list: list[DataProtocol] = []
        self._load_data_paths()

    @property
    def data_path_list(self) -> Iterable[str]:
        return [data.data_path for data in self.data_list]

    # Queries
    def query_path_name(self, query_path: str) -> Iterable[DataProtocol]:
        return list(filter(lambda d: query_path in d.data_path, self.data_list))

    # DataManager Representation
    def tree(self) -> None:  # pragma: no cover
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
            print(" " * 4 + "   └── " + data.data_path)

    def _load_data_paths(self) -> None:
        """
        Create data objects from the data three.
        """
        # From the path get the data path list
        data_path_list = self._get_experiment_paths()

        # Create data object
        self.data_list = []
        invalid_count = 0
        counter = 0
        for counter, path in enumerate(data_path_list):
            data = Data(path)
            if data.check_path_validity():
                self.data_list.append(data)
            else:
                invalid_count += 1
        logging.info(
            f"Total {counter} recording found. There are {invalid_count} invalid paths."
        )

    def _get_experiment_paths(self, sort: bool = True) -> Iterable[str]:
        """
        Get experiment paths.

        Parameters
        ----------
        sort : bool
            Sort the output data paths

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
        if sort:
            if os.name == "nt":  # For Windows
                pattern = r"(\d+)\\recording(\d+)"
            else:  # For Linux or other POSIX systems
                pattern = r"(\d+)/recording(\d+)"
            matches = [re.search(pattern, path) for path in path_list]
            tags: list[tuple[int, int]] = []
            for match in matches:
                if match is not None:
                    tags.append((int(match.group(1)), int(match.group(2))))
            path_list = [path for _, path in sorted(zip(tags, path_list, strict=False))]
        return path_list

    # MutableSequence abstract methods
    def __len__(self):  # type: ignore[no-untyped-def]
        return len(self.data_list)

    def __getitem__(self, idx):  # type: ignore[no-untyped-def]
        return self.data_list[idx]

    def __delitem__(self, idx):  # type: ignore[no-untyped-def]
        del self.data_list[idx]

    def __setitem__(self, idx, data):  # type: ignore[no-untyped-def]
        if data.check_path_validity():
            self.data_list[idx] = data
        else:
            logging.warning("Invalid data cannot be loaded to the DataManager.")

    def insert(self, idx, data):  # type: ignore[no-untyped-def]
        if data.check_path_validity():
            self.data_list.insert(idx, data)
        else:
            logging.warning("Invalid data cannot be loaded to the DataManager.")
