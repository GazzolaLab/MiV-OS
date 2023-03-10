__doc__ = """

Module (Intan)
##################

.. autoclass:: DataIntan
   :members:

.. autoclass:: DataIntanTriggered
   :members:

"""
__all__ = ["DataIntan", "DataIntanTriggered"]

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
from miv.core.datatype import Signal, Spikestamps
from miv.core.wrapper import wrap_cacher
from miv.io.openephys.data import Data, DataManager
from miv.typing import SignalType


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
        sampling_rate = None
        for data in self._generator_by_channel_name("stim_data", progress_bar):
            signals.append(data.data)
            timestamps.append(data.timestamps)
            sampling_rate = data.rate

        return Signal(
            data=np.concatenate(signals, axis=0),
            timestamps=np.concatenate(timestamps),
            rate=sampling_rate,
        )

    def _generator_by_channel_name(self, name: str, progress_bar: bool = False):
        if not self.check_path_validity():
            raise FileNotFoundError("Data directory does not have all necessary files.")
        files = self.get_recording_files()
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
        paths = glob(os.path.join(self.data_path, "*.rhs"), recursive=True)
        paths.sort()
        return paths

    # Disable
    def load_ttl_event(self):
        raise AttributeError("DataIntan does not have laod_ttl_event method")


class DataIntanTriggered(DataIntan):
    """
    DataIntanTriggered is a subclass of DataIntan, which is used to handle Intan
    recording when the recording is triggered by TTL signals.
    """

    def __init__(
        self,
        data_path,  # FIXME: argument order with intan.DATA
        index: int = 0,
        trigger_key: str = "board_adc_data",
        trigger_index: int = 0,
        trigger_threshold_voltage=1.0,
        *args,
        **kwargs,
    ):
        super().__init__(data_path=data_path, *args, **kwargs)
        self.index = index
        self.trigger_key = trigger_key
        self.trigger_index = trigger_index
        self.trigger_threshold_voltage = trigger_threshold_voltage

    def __getitem__(self, index):
        return DataIntanTriggered(
            data_path=self.data_path,
            index=index,
            trigger_key=self.trigger_key,
            trigger_index=self.trigger_index,
            trigger_threshold_voltage=self.trigger_threshold_voltage,
        )

    @wrap_cacher(cache_tag="trigger_grouping")
    def _trigger_grouping(self, paths):
        def _find_sequence(arr):
            if arr.size == 0:
                return arr
            return arr[np.concatenate([np.array([True]), arr[1:] - 1 != arr[:-1]])]

        group_files = []
        group = {"paths": [], "start index": [], "end index": []}
        status = 0
        for file in paths:
            result, _ = rhs.load_file(file)
            time = result["t"]
            adc_data = result[self.trigger_key][self.trigger_index]
            diff_adc_data = adc_data[1:] - adc_data[:-1]
            # TODO: Something can be done better
            recording_on = _find_sequence(
                np.where(diff_adc_data > self.trigger_threshold_voltage)[0]
            ).tolist()
            recording_off = _find_sequence(
                np.where(diff_adc_data < -self.trigger_threshold_voltage)[0]
            ).tolist()

            sindex = 0
            if status == 1 and len(recording_on) == 0 and len(recording_off) == 0:
                group["paths"].append(file)
                group["start index"].append(sindex)
                group["end index"].append(time.shape[0])
            while len(recording_on) > 0 or len(recording_off) > 0:
                if status == 0 and len(recording_on) > 0:
                    sindex = recording_on.pop(0)
                    status = 1
                    if len(recording_off) == 0:
                        group["paths"].append(file)
                        group["start index"].append(sindex)
                        group["end index"].append(time.shape[0])
                elif status == 1 and len(recording_off) > 0:
                    eindex = recording_off.pop(0)
                    status = 0
                    if eindex < sindex:
                        raise ValueError(
                            f"Something went wrong with the trigger signal. Starting index {sindex} must be less than ending index {eindex}."
                        )
                    group["paths"].append(file)
                    group["start index"].append(sindex)
                    group["end index"].append(eindex)
                    group_files.append(group)
                    group = {"paths": [], "start index": [], "end index": []}
                else:
                    raise ValueError(
                        f"Something went wrong with the trigger signal. {len(recording_on)=} {len(recording_off)=}"
                    )

        return group_files

    def get_recording_files(self):
        paths = DataIntan.get_recording_files(self)
        groups = self._trigger_grouping(paths)
        return groups[self.index]["paths"]

    def _generator_by_channel_name(self, name: str, progress_bar: bool = False):
        if not self.check_path_validity():
            raise FileNotFoundError("Data directory does not have all necessary files.")
        groups = self._trigger_grouping(None)  # Should be cached
        files = groups[self.index]["paths"]
        sindex = groups[self.index]["start index"]
        eindex = groups[self.index]["end index"]
        # Get sampling rate from setting file
        setting_path = os.path.join(self.data_path, "settings.xml")
        sampling_rate = int(ET.parse(setting_path).getroot().attrib["SampleRateHertz"])
        # Read each files
        for filename, sidx, eidx in tqdm(
            zip(files, sindex, eindex), disable=not progress_bar
        ):
            result, data_present = rhs.load_file(filename)
            assert data_present, f"Data does not present: {filename=}."
            assert not hasattr(result, name), f"No {name} in the file ({filename=})."

            # signal_group = result["amplifier_channels"]
            yield Signal(
                data=np.asarray(result[name]).T[sidx:eidx, :],
                timestamps=np.asarray(result["t"])[sidx:eidx],
                rate=sampling_rate,
            )

    def get_stimulation_events(self):  # TODO: refactor
        minimum_stimulation_length = 0.010
        data = self.get_stimulation()
        stim = data.data
        timestamps = data.timestamps
        # sampling_rate = data.rate
        stimulated_channels = np.where(np.abs(stim).sum(axis=0))[0]
        if len(stimulated_channels) == 0:
            return None
        stimulated_channel = stimulated_channels[0]
        stim = stim[:, stimulated_channel]

        events = ~np.isclose(stim, 0)
        eventstrain = timestamps[np.where(events)[0]]
        ref = np.concatenate(
            [[True], np.diff(eventstrain) > minimum_stimulation_length]
        )
        eventstrain = eventstrain[ref]
        ret = Spikestamps([eventstrain])  # TODO: use datatype.Events
        return ret
