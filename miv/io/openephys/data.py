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


from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import logging
import os
import re
import pickle
from collections.abc import MutableSequence
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from miv.core.datatype import Signal
from miv.core.operator import DataLoaderMixin
from miv.io.openephys.binary import load_continuous_data, load_recording, load_ttl_event
from miv.io.protocol import DataProtocol
from miv.signal.filter.protocol import FilterProtocol
from miv.signal.spike.protocol import SpikeDetectionProtocol
from miv.statistics import firing_rates
from miv.typing import SignalType

if TYPE_CHECKING:
    import pathlib

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
    ):
        self.data_path: str = data_path
        super().__init__()
        self._analysis_path: str = os.path.join(data_path, "analysis")
        self.masking_channel_set: set[int] = set()

        self.tag: str = f"{tag}"

        os.makedirs(self._analysis_path, exist_ok=True)

    @property
    def analysis_path(self):
        """Default sub-directory path to save analysis results"""
        return self._analysis_path

    @analysis_path.setter
    def analysis_path(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        self._analysis_path = path

    def save_figure(
        self,
        figure: plt.Figure,
        group: str,
        filename: str,
        savefig_kwargs: dict[Any, Any] | None = None,
    ):
        """Save figure in analysis sub-directory

        Parameters
        ----------
        figure : plt.Figure
        group : str
        filename : str
        savefig_kwargs : Optional[Dict[Any, Any]]
            Additional parameters to pass to `plt.savefig`.
        """
        if savefig_kwargs is None:
            savefig_kwargs = {}

        dirpath = os.path.join(self.analysis_path, group)
        os.makedirs(dirpath, exist_ok=True)

        filepath = os.path.join(dirpath, filename)
        plt.figure(figure)
        plt.savefig(filepath, **savefig_kwargs)

    def has_data(self, filename: str):
        """Check if the analysis data already saved

        Parameters
        ----------
        filename : str
            File name to check
        """
        filepath = os.path.join(self.analysis_path, filename + ".pkl")
        return os.path.exists(filepath)

    def save_data(
        self,
        data,
        filename: str,
        pkl_kwargs: dict[Any, any] | None = None,
    ):
        """Save analysis data into sub-directory

        Parameters
        ----------
        data :
        filename : str
        pkl_kwargs : Optional[Dict[Any, Any]]
            Additional parameters to pass to `plt.savefig`.
        """
        pkl_kwargs = pkl_kwargs or {}
        filepath = os.path.join(self.analysis_path, filename + ".pkl")
        with open(filepath, "wb") as output_file:
            pickle.dump(data, output_file, **pkl_kwargs)

    def load_data(
        self,
        filename: str,
        pkl_kwargs: dict[Any, any] | None = None,
    ):
        """Quick load pickled data (data saved using `save_data`)

        Parameters
        ----------
        filename : str
        pkl_kwargs : Optional[Dict[Any, Any]]
            Additional parameters to pass to `plt.savefig`.
        """
        pkl_kwargs = pkl_kwargs or {}
        filepath = os.path.join(self.analysis_path, filename + ".pkl")
        with open(filepath, "rb") as output_file:
            data = pickle.load(output_file, **pkl_kwargs)
        return data

    def load(
        self,
        num_fragments: int | None = None,
        start_at_zero: bool = False,
        progress_bar=False,
        mpi_comm: mpi4py.MPI.Intercomm | None = None,
    ):
        """
        Iterator to load data fragmentally.

        Parameters
        ----------
        num_fragments : Optional[int]
            Instead of loading entire data at once, split the data into `num_fragment`
            number of subdata to process separately. (default=None)
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

        start_index = None
        end_index = None
        if mpi_comm is not None:
            rank = mpi_comm.Get_rank()
            size = mpi_comm.Get_size()
            tasks = np.array_split(np.arange(num_fragments), size)[rank]
            start_index = tasks.min()
            end_index = tasks.max() + 1
        for signal, timestamps, rate in load_recording(
            self.data_path,
            self.masking_channel_set,
            start_at_zero=start_at_zero,
            num_fragments=num_fragments,
            start_index=start_index,
            end_index=end_index,
            progress_bar=progress_bar,
        ):
            yield Signal(data=signal, timestamps=timestamps, rate=rate)

    def load_ttl_event(self):
        """
        Load TTL event data if data contains. Detail implementation is :func:`here <miv.io.binary.load_ttl_event>`.
        """
        states, full_words, timestamps, sampling_rate, initial_state = load_ttl_event(
            self.data_path
        )
        return Signal(data=states[:, None], timestamps=timestamps, rate=sampling_rate)

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

    def clear_channel_mask(self):
        """
        Clears all present channel masks.
        """

        self.masking_channel_set = set()

    def _auto_channel_mask_with_correlation_matrix(
        self,
        spontaneous_binned: dict[str, Any],
        filter: FilterProtocol,
        detector: SpikeDetectionProtocol,
        offset: float = 0,
        bins_per_second: float = 100,
    ):
        """
        Automatically apply mask.

        Parameters
        ----------
        spontaneous_binned : Union[Iterable[Iterable[int]], int]
            [0]: 2D matrix with each column being the binned number of spikes from each channel.
            [1]: number of bins from spontaneous recording binned matrix
            [2]: array of indices of empty channels
        filter : FilterProtocol
            Filter that is applied to the signal before masking.
        detector : SpikeDetectionProtocol
            Spike detector that extracts spikes from the signals.
        offset : float, optional
            The trimmed time in seconds at the front of the signal (default = 0).
        bins_per_second : float, default=100
            Optional parameter for binning spikes with respect to time.
            The spikes are binned for comparison between the spontaneous recording and
            the other experiments. This value should be adjusted based on the firing rate.
            A high value reduces type I error; a low value reduces type II error.
            As long as this value is within a reasonable range, it should negligibly affect
            the result (see jupyter notebook demo).
        """

        exp_binned = self._get_binned_matrix(filter, detector, offset, bins_per_second)
        num_channels = np.shape(exp_binned["matrix"])[1]

        # if experiment is longer than spontaneous recording, it gets trunkated
        if exp_binned["num_bins"] > spontaneous_binned["num_bins"]:
            spontaneous_matrix = spontaneous_binned["matrix"].copy()
            exp_binned["matrix"] = exp_binned["matrix"][
                : spontaneous_binned["num_bins"] + 1
            ]

        # if spontaneous is longer than experiment recording
        elif exp_binned["num_bins"] < spontaneous_binned["num_bins"]:
            spontaneous_matrix = spontaneous_binned["matrix"].copy()
            spontaneous_matrix = spontaneous_matrix[: exp_binned["num_bins"] + 1]

        # they're the same size
        else:
            spontaneous_matrix = spontaneous_binned["matrix"].copy()

        exp_binned_channel_rows = np.transpose(exp_binned["matrix"])
        spontaneous_binned_channel_rows = np.transpose(spontaneous_matrix)

        dot_products = []
        for chan in range(num_channels):
            try:
                dot_products.append(
                    np.dot(
                        spontaneous_binned_channel_rows[chan],
                        exp_binned_channel_rows[chan],
                    )
                )
            except Exception:
                raise Exception(
                    "Number of channels does not match between this experiment and referenced spontaneous recording."
                )

        mean = np.mean(dot_products)
        threshold = mean + np.std(dot_products)

        mask_list = []
        for chan in range(num_channels):
            if dot_products[chan] > threshold:
                mask_list.append(chan)
        self.set_channel_mask(np.concatenate((mask_list, exp_binned["empty_channels"])))

    def _get_binned_matrix(
        self,
        filter: FilterProtocol,
        detector: SpikeDetectionProtocol,
        offset: float = 0,
        bins_per_second: float = 100,
    ) -> dict[str, Any]:
        """
        Performs spike detection and return a binned 2D matrix with columns being the
        binned number of spikes from each channel.

        Parameters
        ----------
        filter : FilterProtocol
            Filter that is applied to the signal before masking.
        detector : SpikeDetectionProtocol
            Spike detector that extracts spikes from the signals.
        offset : float, optional
            The time in seconds to be trimmed in front (default = 0).
        bins_per_second : float, default=100
            Optional parameter for binning spikes with respect to time.
            The spikes are binned for comparison between the spontaneous recording and
            the other experiments. This value should be adjusted based on the firing rate.
            A high value reduces type I error; a low value reduces type II error.
            As long as this value is within a reasonable range, it should negligibly affect
            the result (see jupyter notebook demo).

        Returns
        -------
        matrix :
            2D list with columns as channels.
        num_bins : int
            The number of bins.
        empty_channels : List[int]
            List of indices of empty channels
        """

        result = []
        for sig, times, samp in self.load(num_fragments=1):
            start_time = times[0] + offset
            starting_index = int(offset * samp)

            trimmed_signal = sig[starting_index:]
            trimmed_times = times[starting_index:]

            filtered_sig = filter(trimmed_signal, samp)
            spiketrains = detector(filtered_sig, trimmed_times, samp)

            bins_array = np.arange(
                start=start_time, stop=trimmed_times[-1], step=1 / bins_per_second
            )
            num_bins = len(bins_array)
            num_channels = len(spiketrains)
            empty_channels = []

            for chan in range(num_channels):
                if len(spiketrains[chan]) == 0:
                    empty_channels.append(chan)

                spike_counts = np.zeros(shape=num_bins + 1, dtype=int)
                digitized_indices = np.digitize(spiketrains[chan], bins_array)
                for bin_index in digitized_indices:
                    spike_counts[bin_index] += 1
                result.append(spike_counts)

        return {
            "matrix": np.transpose(result),
            "num_bins": num_bins,
            "empty_channels": empty_channels,
        }

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
        super().__init__()
        self.data_collection_path: str | pathlib.Path = data_collection_path

        # From the path get data paths and create data objects
        self.data_list: list[DataProtocol] = []
        self._load_data_paths()

    def __call__(self):
        pass

    def run(self) -> Signal:
        pass

    @property
    def data_path_list(self) -> Iterable[str]:
        return [data.data_path for data in self.data_list]

    # Queries
    def query_path_name(self, query_path) -> Iterable[DataProtocol]:
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
            pattern = r"(\d+)\/recording(\d+)"  # TODO: probably need better pattern
            matches = [re.search(pattern, path) for path in path_list]
            tags = [(int(match.group(1)), int(match.group(2))) for match in matches]
            path_list = [path for _, path in sorted(zip(tags, path_list))]
        return path_list

    def save(self, tag: str, format: str):  # pragma: no cover
        raise NotImplementedError  # TODO
        for data in self.data_list:
            data.save(tag, format)

    def apply_filter(self, filter: FilterProtocol):  # pragma: no cover
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

    def auto_channel_mask_with_firing_rate(
        self,
        filter: FilterProtocol,
        detector: SpikeDetectionProtocol,
        no_spike_threshold: float = 1,
    ):
        """
        Perform automatic channel masking.
        This method simply applies a Butterworth filter, extract spikes, and filter out
        the channels that contain either no spikes or too many spikes.

        Parameters
        ----------
        filter : FilterProtocol
            Filter that is applied to the signals before detecting spikes.
        detector : SpikeDetectionProtocol
            Spike detector that is used to extract spikes from the filtered signal.
        no_spike_threshold : float, default=1
            Spike rate threshold (spike per sec) for filtering channels with no spikes.
            (default = 1)

        """

        for data in self.data_list:
            for sig, times, samp in data.load(num_fragments=1):
                mask_list = []

                filtered_signal = filter(sig, samp)
                spiketrains = detector(filtered_signal, times, samp)
                spike_stats = firing_rates(spiketrains)

                for idx, channel_rate in enumerate(spike_stats["rates"]):
                    if int(channel_rate) <= no_spike_threshold:
                        mask_list.append(idx)

                data.set_channel_mask(mask_list)

    def auto_channel_mask_with_correlation_matrix(
        self,
        spontaneous_data: DataProtocol,
        filter: FilterProtocol,
        detector: SpikeDetectionProtocol,
        omit_experiments: Iterable[int] | None = None,
        spontaneous_offset: float = 0,
        exp_offsets: Iterable[float] | None = None,
        bins_per_second: float = 100,
    ):
        """
        This masking method uses a correlation matrix between a spontaneous recording and
        the experiment recordings to decide which channels to mask out.

        Notes
        -----
            Sample rate and number of channels for all recordings must be the same

        Parameters
        ----------
        spontaneous_data : Data
            Data from spontaneous recording that is used for comparison.
        filter : FilterProtocol
            Filter that is applied to the signals before detecting spikes.
        detector : SpikeDetectionProtocol
            Spike detector that is used to extract spikes from the filtered signal.
        omit_experiments: Optional[Iterable[int]]
            Integer array of experiment indices (0-based) to omit.
        spontaneous_offset: float, optional
            Postive time offset for the spontaneous experiment (default = 0).
            A negative value will be converted to 0.
        exp_offsets: Optional[Iterable[float]]
            Positive float array of time offsets for each experiment (default = 0).
            Negative values will be converted to 0.
        bins_per_second : float, default=100
            Optional parameter for binning spikes with respect to time.
            The spikes are binned for comparison between the spontaneous recording and
            the other experiments. This value should be adjusted based on the firing rate.
            A high value reduces type I error; a low value reduces type II error.
            As long as this value is within a reasonable range, it should negligibly affect
            the result (see jupyter notebook demo).
        """

        omit_experiments_list: list[float] = (
            list(omit_experiments) if omit_experiments else []
        )
        exp_offsets_list: list[float] = list(exp_offsets) if exp_offsets else []

        if spontaneous_offset < 0:
            spontaneous_offset = 0

        exp_offsets_length = sum(1 for e in exp_offsets_list)
        for i in range(exp_offsets_length):
            if exp_offsets_list[i] < 0:
                exp_offsets_list[i] = 0

        if exp_offsets_length < len(self.data_list):
            exp_offsets_list = np.concatenate(
                (
                    np.array(exp_offsets_list),
                    np.zeros(len(self.data_list) - exp_offsets_length),
                )
            )

        spontaneous_binned = spontaneous_data._get_binned_matrix(
            filter, detector, spontaneous_offset, bins_per_second
        )

        for exp_index, data in enumerate(self.data_list):
            if not (exp_index in omit_experiments_list):
                data._auto_channel_mask_with_correlation_matrix(
                    spontaneous_binned,
                    filter,
                    detector,
                    exp_offsets_list[exp_index],
                    bins_per_second,
                )
