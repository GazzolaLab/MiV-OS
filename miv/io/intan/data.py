__doc__ = """

Module (Intan)
##############

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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import miv.io.intan.rhs as rhs
from miv.core.datatype import Events, Signal, Spikestamps
from miv.core.operator.wrapper import cache_functional
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def export(self, filename, channels=None, progress_bar: bool = False):
        """
        Export data to the specified path.
        TODO: implement "view" time range

        Parameters
        ----------
        filename : str or Path
            Path to export data.
        """
        from miv.io import file as miv_file

        data = miv_file.initialize()
        miv_file.create_group(data, "Ephys", counter="nobj")
        miv_file.create_dataset(data, "Timestamps", group="Ephys", dtype=np.float32)
        miv_file.create_dataset(data, "Rate", group="Ephys", dtype=np.float32)
        miv_file.create_dataset(data, "Data", group="Ephys", dtype=np.float32)

        container = miv_file.create_container(data)
        data_shape = None
        for signal in tqdm(self.load(), disable=not progress_bar):
            if channels is None:
                matrix = signal.data
            else:
                channels = np.asarray(channels)
                matrix = signal.data[:, channels]
            if data_shape is None:
                data_shape = matrix.shape
            elif data_shape != matrix.shape:
                break
            container["Ephys/Data"] = matrix
            container["Ephys/Timestamps"] = signal.timestamps
            container["Ephys/Rate"] = signal.rate
            test = miv_file.pack(data, container)
            assert test == 0

        miv_file.write(filename, data)

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

        active_channels, total = self._get_active_channels()
        for sig in self._generator_by_channel_name("amplifier_data"):
            self._expand_channels(sig, active_channels, total)
            yield sig

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

        signal = Signal(
            data=np.concatenate(signals, axis=0),
            timestamps=np.concatenate(timestamps),
            rate=sampling_rate,
        )
        active_channels, total = self._get_active_channels()
        self._expand_channels(signal, active_channels, total)
        return signal

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

            signal = np.asarray(result[name]).T

            yield Signal(
                data=signal,
                timestamps=np.asarray(result["t"]),
                rate=sampling_rate,
            )

    def _get_active_channels(self, group_prefix=("A", "B", "C", "D")):
        setting_path = os.path.join(self.data_path, "settings.xml")
        root = ET.parse(setting_path).getroot()
        total = 0
        active_channels = []
        for child in root:
            if child.tag != "SignalGroup":
                continue
            if child.attrib["Prefix"] not in group_prefix:
                continue
            for channel in child:
                if channel.attrib["Enabled"] == "True":
                    active_channels.append(total)
                total += 1
        return np.array(active_channels), total

    def _expand_channels(
        self, signal: SignalType, active_channels, num_active_channels: int
    ):
        """
        Expand number of channels in `signal` to match the active channels.
        """
        if num_active_channels != signal.number_of_channels:
            _data = np.zeros(
                [signal.shape[0], num_active_channels], dtype=signal.data.dtype
            )
            if active_channels.size != signal.data.shape[1]:
                # TODO: figure out why this can happen
                self.logger.warning(
                    f"Size mismatch in the data: {active_channels.size} enabled channel vs {signal.data.shape=}"
                )
                if signal.data.shape[1] > active_channels.size:
                    _data[:, active_channels] = signal.data[:, active_channels.size]
                elif signal.data.shape[1] < active_channels.size:
                    _data[:, active_channels[: signal.data.shape[1]]] = signal.data
            else:
                _data[:, active_channels] = signal.data
            signal.data = _data

    def _read_header(self):
        filename = self.get_recording_files()[0]
        fid = open(filename, "rb")
        header = rhs.read_header(fid)
        return header

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
        """
        Get list of path of all recording files.
        """
        paths = glob(os.path.join(self.data_path, "*.rhs"), recursive=True)
        paths.sort()
        return paths

    def get_stimulation_events(self):  # TODO: refactor
        """
        Get stimulation in Spikestamps form, where each stamps represent the stimulus event.
        """
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

    def _load_digital_event_common(self, name, num_channels, progress_bar=False):
        stamps = [[] for _ in range(num_channels)]
        for sig in self._generator_by_channel_name(name, progress_bar=progress_bar):
            for channel in range(num_channels):
                index = np.where(sig.data[:, channel])[0]
                stamps[channel].extend(np.asarray(sig.timestamps)[index])
        return Spikestamps(stamps)

    # @cache_functional(cache_tag="digital_in")
    def load_digital_in_event(
        self,
        progress_bar: bool = False,
    ):  # pragma: no cover
        """
        Load recorded data from digital input ports.
        Result is a list of timestamps for each channel, in Spikestamps format.

        Parameters
        ----------
        progress_bar : bool, optional
            Show progress bar, by default False
        """
        self.header = self._read_header()
        num_channels = self.header["num_board_dig_in_channels"]
        return self._load_digital_event_common(
            "board_dig_in_data", num_channels, progress_bar=progress_bar
        )

    # @cache_functional(cache_tag="digital_out")
    def load_digital_out_event(
        self,
        progress_bar: bool = False,
    ):  # pragma: no cover
        """
        Load recorded data from digital output ports.
        Result is a list of timestamps for each channel, in Spikestamps format.

        Parameters
        ----------
        progress_bar : bool, optional
            Show progress bar, by default False
        """
        self.header = self._read_header()
        num_channels = self.header["num_board_dig_out_channels"]
        return self._load_digital_event_common(
            "board_dig_out_data", num_channels, progress_bar=progress_bar
        )

    # @cache_functional(cache_tag="ttl_events")
    def load_ttl_event(
        self,
        deadtime: float = 0.002,
        compress: bool = False,
        progress_bar: bool = False,
    ):
        """
        Load TTL events recorded data.

        Parameters
        ----------
        deadtime : float
            Deadtime between two TTL events. (default: 0.002)
        compress : bool
            If True, reduce rate of the signal. (default: False)
        progress_bar : bool
            If True, show progress bar. (default: False)
        """

        signals, timestamps = [], []
        sampling_rate = None
        active_channels, total = self._get_active_channels()
        for data in self._generator_by_channel_name("stim_data", progress_bar):
            self._expand_channels(data, active_channels, total)
            dead_time_idx = int(deadtime * data.rate)
            num_channels = data.number_of_channels

            signal = np.zeros((data.shape[data._SIGNALAXIS], 1), dtype=np.int_)
            for channel in range(num_channels):
                array = data.data[:, channel]
                threshold_crossings = np.nonzero(array)[0]
                if len(threshold_crossings) == 0:
                    continue

                distance_sufficient = np.insert(
                    np.diff(threshold_crossings) >= dead_time_idx, 0, True
                )
                while not np.all(distance_sufficient):
                    # repeatedly remove all threshold crossings that violate the dead_time
                    threshold_crossings = threshold_crossings[distance_sufficient]
                    distance_sufficient = np.insert(
                        np.diff(threshold_crossings) >= dead_time_idx, 0, True
                    )
                signal[threshold_crossings, 0] = channel
            signals.append(signal)
            timestamps.append(data.timestamps)
            sampling_rate = data.rate

        data = np.concatenate(signals, axis=0)
        timestamps = np.concatenate(timestamps)

        if compress:  # TODO
            raise NotImplementedError

        return Signal(data=data, timestamps=timestamps, rate=sampling_rate)


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
        progress_bar: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(data_path=data_path, *args, **kwargs)
        self.index = index
        self.trigger_key = trigger_key
        self.trigger_index = trigger_index
        self.trigger_threshold_voltage = trigger_threshold_voltage
        self.progress_bar = progress_bar

    def __len__(self):
        groups = self._trigger_grouping()
        return len(groups)

    def __getitem__(self, index):
        groups = self._trigger_grouping()
        if len(groups) <= index:
            raise IndexError(
                f"Index exceeds the number of triggered recordings ({len(groups)})."
            )
        return DataIntanTriggered(
            data_path=self.data_path,
            index=index,
            trigger_key=self.trigger_key,
            trigger_index=self.trigger_index,
            trigger_threshold_voltage=self.trigger_threshold_voltage,
            progress_bar=self.progress_bar,
        )

    @cache_functional(cache_tag="trigger_grouping")
    def _trigger_grouping(self):
        def _find_sequence(arr):
            if arr.size == 0:
                return arr
            return arr[np.concatenate([np.array([True]), arr[1:] - 1 != arr[:-1]])]

        paths = DataIntan.get_recording_files(self)

        group_files = []
        group = {"paths": [], "start index": [], "end index": []}
        status = 0
        for file in tqdm(paths, disable=not self.progress_bar):
            result, _ = rhs.load_file(file)
            time = result["t"]

            self.logger.info(f"Recorded result keys: {list(result.keys())}")
            self.logger.info(f"Number of trigger: {len(result[self.trigger_key])}")

            # TODO: Something can be done better
            if "adc" in self.trigger_key:
                adc_data = result[self.trigger_key][self.trigger_index]
                diff_adc_data = adc_data[1:] - adc_data[:-1]
                recording_on = _find_sequence(
                    np.where(diff_adc_data > self.trigger_threshold_voltage)[0]
                ).tolist()
                recording_off = _find_sequence(
                    np.where(diff_adc_data < -self.trigger_threshold_voltage)[0]
                ).tolist()
            elif "dig" in self.trigger_key:
                recording_state = result[self.trigger_key][self.trigger_index]
                # recording_state = np.logical_xor(digital_data[1:], digital_data[:-1])
                recording_on = _find_sequence(
                    np.where(
                        np.logical_and(recording_state[1:], ~recording_state[:-1])
                    )[0]
                ).tolist()
                recording_off = _find_sequence(
                    np.where(
                        np.logical_and(~recording_state[1:], recording_state[:-1])
                    )[0]
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
        groups = self._trigger_grouping()
        return groups[self.index]["paths"]

    def _generator_by_channel_name(self, name: str, progress_bar: bool = False):
        # TODO: move out _get_active_channels
        if not self.check_path_validity():
            raise FileNotFoundError("Data directory does not have all necessary files.")
        groups = self._trigger_grouping()
        files = groups[self.index]["paths"]
        sindex = groups[self.index]["start index"]
        eindex = groups[self.index]["end index"]
        # Get sampling rate from setting file
        setting_path = os.path.join(self.data_path, "settings.xml")
        sampling_rate = int(ET.parse(setting_path).getroot().attrib["SampleRateHertz"])
        active_channels, total = self._get_active_channels()
        # Read each files
        for filename, sidx, eidx in tqdm(
            zip(files, sindex, eindex), disable=not progress_bar
        ):
            result, data_present = rhs.load_file(filename)
            assert data_present, f"Data does not present: {filename=}."
            assert not hasattr(result, name), f"No {name} in the file ({filename=})."

            signal = np.asarray(result[name])
            if total != signal.shape[0]:
                _signal = np.zeros([total, signal.shape[1]], dtype=signal.dtype)
                _signal[active_channels, :] = signal
                signal = _signal

            yield Signal(
                data=signal.T[sidx:eidx, :],
                timestamps=np.asarray(result["t"])[sidx:eidx],
                rate=sampling_rate,
            )
