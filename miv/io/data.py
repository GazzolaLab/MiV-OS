from collections.abc import MutableSequence
from typing import Optional
import os
import numpy as np
from miv.signal.filter import FilterProtocol
from miv.typing import SignalType


class Data:
    """
    For each continues.dat file, there will be one Data object
    """

    def __init__(
        self,
        data_path: str,
        channels: int,
        sampling_rate: float = 30000,
        timestamps_npy: Optional[str] = "",
    ):
        self.data_path = data_path
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.timestamps_npy = timestamps_npy

    def load(
        self,
    ):

        """
        Describe function

        Parameters
        ----------
            data_file: continuous.dat file from Open_Ethys recording
            channels: number of recording channels recorded from

        Returns
        -------
            raw_data:
            timestamps:

        """

        raw_data: np.ndarray = np.memmap(self.data_path, dtype="int16")
        length = raw_data.size // self.channels
        raw_data = np.reshape(raw_data, (length, self.channels))

        timestamps_zeroed = np.array(range(0, length)) / self.sampling_rate
        if self.timestamps_npy == "":
            timestamps = timestamps_zeroed
        else:
            timestamps = np.load(self.timestamps_npy) / self.sampling_rate

        # only take first 32 channels
        raw_data = raw_data[:, 0 : self.channels]

        # TODO: do we want timestaps a member of the class?
        return np.array(timestamps), np.array(raw_data)

    def unload(
        self,
    ):
        # TODO: remove the data from memory
        pass

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


class Dataset(MutableSequence):
    def __init__(
        self,
        data_folder_path: str,
        channels: int,
        sampling_rate: float = 30000,
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
