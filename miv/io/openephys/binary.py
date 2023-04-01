__doc__ = """

-------------------------------------

Raw Data Loader
###############

"""
__all__ = [
    "load_continuous_data",
    "load_recording",
    "oebin_read",
    "apply_channel_mask",
    "load_ttl_event",
]

from typing import Any, Dict, List, Optional, Sequence, Set, Union

import logging
import os
from ast import literal_eval
from glob import glob

import neo
import numpy as np
import quantities as pq
from tqdm import tqdm

from miv.typing import SignalType, TimestampsType


def apply_channel_mask(signal: np.ndarray, channel_mask: Set[int]):
    """Apply channel mask on the given signal.

    Parameters
    ----------
    signal : np.ndarray
        Shape of the signal is expected to be (num_data_point, num_channels).
    channel_mask : Set[int]

    Returns
    -------
    output signal : SignalType

    Raises
    ------
    IndexError
        Typically raise index error when the dimension of the signal is less than 2.
    AttributeError
        If signal is non numpy array type.

    """

    num_channels = signal.shape[1]
    channel_index_set = set(range(num_channels)) - channel_mask
    channel_index = np.array(np.sort(list(channel_index_set)))
    signal = signal[:, channel_index]
    return signal


def bits_to_voltage(signal: SignalType, channel_info: Sequence[Dict[str, Any]]):
    """
    Convert binary bit data to voltage (microVolts)

    Parameters
    ----------
    signal : SignalType, numpy array
    channel_info : Dict[str, Dict[str, Any]]
        Channel information dictionary. Typically located in `structure.oebin` file.
        Channel information includes bit-volts conversion ration and units (uV or mV).

    Returns
    -------
    signal : numpy array
        Output signal is in microVolts (uV) unit.

    """
    resultant_unit = pq.Quantity(1, "uV")  # Final Unit
    for channel in range(len(channel_info)):
        bit_to_volt_conversion = channel_info[channel]["bit_volts"]
        recorded_unit = pq.Quantity([1], channel_info[channel]["units"])
        unit_conversion = (recorded_unit / resultant_unit).simplified
        signal[:, channel] *= bit_to_volt_conversion * unit_conversion
    return signal


def oebin_read(file_path: str):
    """
    Oebin file reader in dictionary form

    Parameters
    ----------
    file_path : str

    Returns
    -------
    info : Dict[str, any]
        recording information stored in oebin file.
    """
    # TODO: may need fix for multiple continuous data.
    # TODO: may need to include processor name/id
    info = literal_eval(open(file_path).read())
    return info


def load_ttl_event(
    folder: str,
    return_sample_numbers: bool = False,
):
    """
    Loads TTL event data recorded by Open Ephys as numpy arrays.

    `Reference: OpenEphys TTL data structure <https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html#events>`_

    The path should contain:

    - states.npy: N 16-bit integers, indicating ON/OFF (channel number)
    - sample_numbers.npy: N 64-bit integers, sample number during acquisition.
    - timestamps.npy: N 64-bit floats, global timestamps
    - full_words.npy: N 64-bit integer, TTL word of current state of all lines.

    Extra data are retrieved from:

    - structure.oebin: number of channels and sampling rate.

    Parameters
    ----------
    folder: str
        Folder containing the subfolder 'experiment1'.
    return_sample_numbers: bool
        If set to true, also return sample_numbers that can be used to re-calculate
        synchronization between time series. (default=False)

    Returns
    -------
    states : np.ndarray
        Numpy integer array, indicating ON/OFF state. (+- channel number)
    full_words : np.ndarray
        Numpy integer array, consisting current state of all lines.
    timestamps : TimestampsType
        Numpy float array. Global timestamps in seconds. Relative to start
        of the Record Node's main data stream.
    sampling_rate: float
        Recorded sampling rate
    initial_state: int
        Initial TTL state across lines.
    sample_numbers: Optional[np.ndarray]
        Return if `return_sample_numbers` is true. Return array of sample numbers that
        records sampled clock count. Typically used to synchronize time array.

    Raises
    ------
    AssertionError
        No events recorded in data.

    """

    # Check TTL event recorded
    info_file: str = os.path.join(folder, "structure.oebin")
    info: Dict[str, Any] = oebin_read(info_file)
    version = info["GUI version"]
    assert "events" in info.keys(), "No events recorded (TTL)."
    ttl_info = [
        data
        for data in info["events"]
        if "TTL Input" in data["channel_name"] or "TTL events" in data["channel_name"]
    ]
    assert len(ttl_info) > 0, "No events recorded (TTL)."
    assert (
        len(ttl_info) == 1
    ), "Multiple TTL input is found, which is not supported yet. (TODO)"
    ttl_info = ttl_info[0]

    # Data Structure (OpenEphys Structure)
    v_major, v_minor, v_sub = map(int, version.split("."))
    if v_major == 0 and v_minor <= 5 and v_sub == 4:  # Legacy file name before 0.6.0
        file_states = "channel_states.npy"
        file_timestamps = "timestamps.npy"
        file_sample_numbers = "channels.npy"
        file_full_words = "full_words.npy"
    elif v_major == 0 and v_minor <= 5:  # Legacy file name before 0.6.0
        file_states = "states.npy"
        file_timestamps = "synchronized_timestamps.npy"
        file_sample_numbers = "timestamps.npy"
        file_full_words = "full_words.npy"
    else:
        file_states = "states.npy"
        file_timestamps = "timestamps.npy"
        file_sample_numbers = "sample_numbers.npy"
        file_full_words = "full_words.npy"
    file_path = os.path.join(folder, "events", ttl_info["folder_name"])

    states = np.load(os.path.join(file_path, file_states)).astype(np.int16)
    sample_numbers = np.load(os.path.join(file_path, file_sample_numbers)).astype(
        np.int64
    )
    timestamps = np.load(os.path.join(file_path, file_timestamps)).astype(np.float64)
    full_words = np.load(os.path.join(file_path, file_full_words)).astype(np.int64)

    # Load from structure.oebin file
    sampling_rate: float = ttl_info["sample_rate"]
    initial_state: int = ttl_info["initial_state"] if "initial_state" in ttl_info else 0

    if return_sample_numbers:
        return (
            states,
            full_words,
            timestamps,
            sampling_rate,
            initial_state,
            sample_numbers,
        )
    else:
        return states, full_words, timestamps, sampling_rate, initial_state


def load_recording(
    folder: str,
    channel_mask: Optional[Set[int]] = None,
    start_at_zero: bool = True,
    num_fragments: int = 1,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    dtype: np.dtype = np.float32,
    verbose: bool = False,
    progress_bar: bool = False,
):
    """
    Loads data recorded by Open Ephys in Binary format as numpy memmap.
    The path should contain

    - continuous/<processor name>/continuous.dat: signal (cannot have multiple file)
    - continuous/<processor name>/timestamps.npy: timestamps
    - structure.oebin: number of channels and sampling rate.

    Parameters
    ----------
    folder: str
        folder containing at least the subfolder 'experiment1'.
    channel_mask: Set[int], optional
        Channel index list to ignore in import (default=None)
    start_at_zero : bool
        If True, the timestamps is adjusted to start at zero.
        Note, recorded timestamps might not start at zero for some reason.
    num_fragments : Optional[int]
        Instead of loading entire data at once, split the data into `num_fragment`
        number of subdata to process separately. By default, num_fragments is set
        to split the recording into 1 minute segments. (default=None)
    start_index : Optional[int]
        Start index of the fragments. It is useful when you want to submit MPI processing.
        For example, one can submit num_fragments=10, start_index=3 to process 3-9 fragments.
        (Zero-indexing)
    end_index : Optional[int]
        End index of the fragments. It is useful when you want to submit MPI processing.
        For example, one can submit num_fragments=10, end_index=5 to process 0-5 fragments.
        (Zero-indexing)
    dtype: np.dtype
        If None, skip data-type conversion. If the filesize is too large, it is advisable
        to keep `dtype=None` and convert slice by slice. (default=float32)
    verbose: bool
        Include logging.info messages for intermediate progresses. (default=False)

    Returns
    -------
    signal : SignalType, neo.core.AnalogSignal
    timestamps : TimestampsType
    sampling_rate : float

    Raises
    ------
    AssertionError
        If more than one "continuous.dat" file exist in the directory.

    """

    file_path: List[str] = glob(
        os.path.join(folder, "**", "continuous.dat"), recursive=True
    )
    assert (
        len(file_path) == 1
    ), f"There should be only one 'continuous.dat' file. (There exists {file_path})"

    # load structure information dictionary
    info_file: str = os.path.join(folder, "structure.oebin")
    info: Dict[str, Any] = oebin_read(info_file)
    num_channels: int = info["continuous"][0]["num_channels"]
    sampling_rate: float = float(info["continuous"][0]["sample_rate"])
    # channel_info: Dict[str, Any] = info["continuous"][0]["channels"]

    _old_oe_version = False
    if "GUI version" in info:
        version = info["GUI version"]
        v_major, v_minor, v_sub = list(map(int, version.split(".")))[:3]
        _old_oe_version = v_major == 0 and v_minor <= 5  # Legacy
    signal, timestamps = load_continuous_data(
        file_path[0], num_channels, sampling_rate, _old_oe_version=_old_oe_version
    )
    if num_fragments is None:
        num_fragments = int(max(timestamps.shape[0] // sampling_rate // (60 + 1), 1))
    fragmented_signal = np.array_split(signal, num_fragments, axis=0)[
        start_index:end_index
    ]
    fragmented_timestamps = np.array_split(timestamps, num_fragments)[
        start_index:end_index
    ]
    for frag_id in tqdm(range(num_fragments), disable=not progress_bar):
        _signal = fragmented_signal[frag_id]
        _timestamps = fragmented_timestamps[frag_id]
        if _signal.shape[0] > 6.0e6:
            logging.warn(
                "The recording data is too long. Consider using more num_fragments."
            )
        _signal = _signal.astype(dtype)
        if verbose:
            logging.info("Cast done")

        # To Voltage
        _signal = bits_to_voltage(_signal, info["continuous"][0]["channels"])

        if channel_mask:
            _signal = apply_channel_mask(_signal, channel_mask)

        # Adjust timestamps to start from zero
        if start_at_zero and not np.isclose(_timestamps[0], 0.0):
            _timestamps -= _timestamps[0]

        yield _signal, _timestamps, sampling_rate


def load_continuous_data(
    data_path: str,
    num_channels: int,
    sampling_rate: float,
    timestamps_path: Optional[str] = None,
    _recorded_dtype: Union[np.dtype, str] = "int16",
    _old_oe_version: bool = False,
):
    """
    Load single continous data file and return timestamps and raw data in numpy array.
    Typical `data_path` from OpenEphys has a name `continuous.dat`.

    .. note::
        The output data is raw-data without unit conversion. In order to convert the unit
        to voltage, you need to multiply by `bit_volts` conversion ratio. This ratio and
        units are typially saved in `structure.oebin` file.

    Parameters
    ----------
    data_path : str
        continuous.dat file path from Open_Ethys recording.
    num_channels : int
        number of recording channels recorded. Note, this method will not throw an error
        if you don't provide the correct number of channels.
    sampling_rate : float
        data sampling rate.
    timestamps_path : Optional[str]
        If None, first check if the file "timestamps.npy" exists on the same directory.
        If the file doesn't exist, we deduce the timestamps based on the sampling rate
        and the length of the data.
    _recorded_dtype: Union[np.dtype, str]
        Recorded data type. (default="int16")

    Returns
    -------
    raw_data: SignalType, numpy array
    timestamps: TimestampsType, numpy array

    Raises
    ------
    FileNotFoundError
        If data_path is invalid.
    ValueError
        If the error message shows the array cannot be reshaped due to shape,
        make sure the num_channels is set accurately.

    """

    # Read raw data signal
    raw_data: np.ndarray = np.memmap(
        data_path, dtype=_recorded_dtype, mode="r", order="C"
    )
    length = raw_data.size // num_channels
    raw_data = raw_data.reshape(length, num_channels)

    # Get timestamps_path
    if timestamps_path is None:
        dirname = os.path.dirname(data_path)
        timestamps_path = os.path.join(dirname, "timestamps.npy")

    # Get timestamps
    if os.path.exists(timestamps_path):
        timestamps = np.asarray(
            np.load(timestamps_path)
        )  # TODO: check if npy file includes dtype. else, add "dtype=np.float32"
        if _old_oe_version:
            timestamps = timestamps / float(sampling_rate)
    else:  # If timestamps_path doesn't exist, deduce the stamps
        timestamps = np.array(range(0, length)) / sampling_rate

    return raw_data, timestamps
