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

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
from typing import Any, Callable, Iterable, List, Optional, Set
||||||| parent of 2f9efba (update: analysis figure save)
from typing import Any, Optional, Iterable, Callable, List, Set
=======
from typing import Any, Optional, Iterable, Callable, List, Set, Dict
>>>>>>> 2f9efba (update: analysis figure save)
||||||| parent of 63a4820 (add channel mask modifying functions in data.py and implement auto channel mask with no-spike channels)
from typing import Any, Callable, Dict, Iterable, List, Optional, Set
=======
||||||| parent of 6127f95 (fix for loop index error)
=======
from asyncio.windows_events import NULL
<<<<<<< HEAD
>>>>>>> 6127f95 (fix for loop index error)
from sqlite3 import Timestamp
import statistics
||||||| parent of 058da41 (:construction: change case convention, set filter and detector as parameters, change oneOverBinSize to bins_per_second)
from sqlite3 import Timestamp
import statistics
=======
>>>>>>> 058da41 (:construction: change case convention, set filter and detector as parameters, change oneOverBinSize to bins_per_second)
from typing import Any, Callable, Dict, Iterable, List, Optional, Set
>>>>>>> 63a4820 (add channel mask modifying functions in data.py and implement auto channel mask with no-spike channels)

import logging
import os
from collections.abc import MutableSequence
from contextlib import contextmanager
from glob import glob

import numpy as np
from scipy.fft import fft, ifft

import matplotlib.pyplot as plt

from miv.io.binary import load_continuous_data, load_recording
from miv.signal.filter.protocol import FilterProtocol
from miv.signal.spike.protocol import SpikeDetectionProtocol
from miv.signal.spike import ThresholdCutoff
from miv.statistics import spikestamps_statistics
from miv.typing import SignalType
from miv.signal.filter import ButterBandpass

import elephant
import neo

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

    
    def clear_channel_mask(self):
        """
        Clears all present channel masks.
        """

        self.masking_channel_set.update([])

        
    def add_channel_mask(self, channel_id: Iterable[int]):
        """
        Put mask on more channels.

        Parameters
        ----------
        channel_id : Iterable[int], list
            List of channel id that will be added to the mask
        """
        self.masking_channel_set.update(self.masking_channel_set.union(channel_id))





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


    def auto_channel_mask_baseline(self, no_spike_threshold: float = 1, constant_spike_threshold: float = 20):
        """
        Perform automatic channel masking.

        Parameters
        ----------
        no_spike_threshold : float
            spike rate threshold (spike per sec) for filtering channels with no spikes
        constant_spike_threshold : float
            spike rate threshold (spike per sec) for filtering channels with constant spikes
        
        """
        # 1. Channels with no spikes should be masked
        # 2. Channels with constant spikes shoudl be masked

        detector = ThresholdCutoff()
        for data in self.data_list:
            with data.load() as (sig, times, samp):
                noSpikeChannelList : list[int] = []
                spiketrains = detector(sig, times, samp)
                spiketrainsStats = spikestamps_statistics(spiketrains)
                
                for channel in range(len(spiketrainsStats['rates'])):
                    channelSpikeRate = spiketrainsStats['rates'][channel]

                    if channelSpikeRate < no_spike_threshold or channelSpikeRate > constant_spike_threshold:
                        noSpikeChannelList.append(channel)
                
                data.add_channel_mask(noSpikeChannelList)


    def auto_channel_mask_v1(self, no_spike_threshold: float = 0.01, isiThreshold: float = 0.1):
        """
        Perform automatic channel masking.

        Parameters
        ----------
        no_spike_threshold : float
            Spike rate threshold (spike per sec) for filtering channels with no spikes

        isiThreshold : float
            Inter-spike-interval threshold (seconds) for filtering channels with constant spikes.
            Channels with ISI median less than this value will be masked.
        
        """
        # 1. Channels with no spikes should be masked
        # 2. Channels with constant spikes should be masked

        detector = ThresholdCutoff()
        for data in self.data_list:
            
            with data.load() as (sig, times, samp):
                maskList : list[int] = []
                spiketrains = detector(sig, times, samp)

                for channel in range(len(spiketrains)):
                    channelISI = elephant.statistics.isi(spiketrains[channel]).__array__()
                    
                    # determining channels with no spikes
                    numSpikesThreshold = no_spike_threshold*(len(times)/samp)
                    if (len(channelISI) < numSpikesThreshold):
                        maskList.append(channel)

                    # determining channels with constant spikes
                    else:
                        if np.median(channelISI) < isiThreshold:
                            maskList.append(channel)
                
                data.add_channel_mask(maskList)
                print(maskList)
                print(numSpikesThreshold)


    def auto_channel_mask_v2(self, compressionPercentage: float = 0.008):
        """
        Perform automatic channel masking.
        First uses FFT to compress the signal, then statistically pick out noisy channels

        Parameters
        ----------
        compressionPrecentage : float
            Compression factor for FFT
        """

        for data in self.data_list:

            with data.load() as (sig, times, samp):
                sig = sig.transpose()
                compressedSignals = []
                
                for channel in range(len(sig)):
                    channelDFT = fft(sig[channel])
                    channelDFTAbs = abs(channelDFT)
                    cutoffCoef = np.percentile(channelDFTAbs, 100*(1-compressionPercentage))
                    
                    # Zero out the smaller coefficient terms
                    for i in range(len(channelDFT)):
                        if (channelDFTAbs[i] < cutoffCoef):
                            channelDFT[i] = 0

                    compressedSignals.append(np.absolute(ifft(channelDFT)))


    def auto_channel_mask_v3(self, comparisonThreshold: float = 3, cleanBenchmark: int = 1):
        """
        This version aims to compare the signals from the spontaneous experiment
        with another recordeing to determine which channels to mask.

        Parameters
        ----------
        comparisonThreshold : float
            The threshold for comparison for the firing rate in each channel between the two experiments

        cleanBenchmark : int
            The index of the comparison experiment
        """

        detector = ThresholdCutoff()
        maskList = []

        with self.data_list[0].load() as (sig, times, samp):
            spiketrains = detector(sig, times, samp)
            spontaneousRates = spikestamps_statistics(spiketrains)['rates']

        with self.data_list[cleanBenchmark].load() as (sig, times, samp):
            spiketrains = detector(sig, times, samp)
            benchmarkRates = spikestamps_statistics(spiketrains)['rates']
        
        for channel in range(len(spontaneousRates)):
            if (benchmarkRates < comparisonThreshold*spontaneousRates[channel]):
                maskList.append(channel)
        
        for data in self.data_list:
            data.add_channel_mask(maskList)

    
    def auto_channel_mask_v4(self, 
                             filter: FilterProtocol, 
                             detector: SpikeDetectionProtocol, 
                             bins_per_second: float = 1000):
        """
        This version attempts to use a correlation matrix to figure out how significant
        each channel is, compared to the spontaneous experiment.
        ** Experiment 1 is the spontaneous one **
        
        Parameters
        ----------
        filter : FilterProtocol
            Filter that is applied to the signals before detecting spikes.
        detector : SpikeDetectionProtocol
            Spike detector that is used to extract spikes from the filtered signal. 
        bins_per_second : float
            Parameter for binning spikes with respect to time.
            The spikes are binned for comparison between the spontaneous recording and 
            the other experiments. This value should be adjusted based on the firing rate.
            A high value reduces type I error; a low value reduces type II error.
        """


        # This section obtains the first half of the matrix used for the correlation matrix
        # Each column is a channel in the spontaneous experiment
        # Each row is number of spikes for each bin
        spontaneous_binned = []
        with self.data_list[0].load() as (sig, times, samp):
            filtered_sig = filter(sig, samp)
            spontaneous_spiketrains = detector(filtered_sig, times, samp)
            num_bins = int(bins_per_second * (times[-1] - times[0]))
            bins = np.arange(start=0, stop=times[-1], step=1/(bins_per_second/num_bins))
            num_channels = len(spontaneous_spiketrains)

            for chan in range(num_channels):
                spike_counts = np.zeros(shape=[int(num_bins)+1], dtype=int)
                bin_indices = np.digitize(spontaneous_spiketrains[chan], bins)
                
                for spike_index in range(len(bin_indices)):
                    spike_counts[bin_indices[spike_index]] += 1
                spontaneous_binned.append(spike_counts)
        spontaneous_binned = np.transpose(spontaneous_binned)

        # This section iterates through each other experiment and calculates correlation matrix
        for exp in range(1, len(self.data_list)):
            experiment_binned = []
            mask_list = []

            with self.data_list[exp].load() as (sig, times, samp):
                filtered_sig = filter(sig, samp)
                experiment_spiketrains = detector(filtered_sig, times, samp)

                for chan in range(num_channels):
                    # filter out empty channels
                    if (len(experiment_spiketrains) == 0):
                        mask_list.append(chan)

                    spike_counts = np.zeros(shape=[int(num_bins)+1], dtype=int)
                    bin_indices = np.digitize(experiment_spiketrains[chan], bins)

                    for i in range(len(bin_indices)):
                        spike_counts[bin_indices[i]] += 1
                    experiment_binned.append(spike_counts)
            experiment_binned = np.transpose(experiment_binned)

            # correlation matrix
            correlation_matrix = np.concatenate((spontaneous_binned, experiment_binned), axis=1)
            correlation_matrix = np.matmul(np.transpose(correlation_matrix), correlation_matrix)
            dot_products = []
            for chan in range(num_channels):
                dot_products.append(correlation_matrix[chan][chan+num_channels])
            mean = np.mean(dot_products)
            for chan in range(len(dot_products)):
                if (dot_products[chan] > mean):
                    mask_list.append(chan)

            self.data_list[exp].add_channel_mask(mask_list)