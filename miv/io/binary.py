__doc__ = """

-------------------------------------

Raw Data Loader
###############

"""
__all__ = ["load_continuous_data", "load_recording", "oebin_read", "apply_channel_mask"]

from typing import Any, Dict, List, Optional, Sequence, Set, Union

import os
from ast import literal_eval
from glob import glob

import neo
import numpy as np
import quantities as pq

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
        channel information includes bit-volts conversion ration and units (uV or mV).

    Returns
    -------
    signal : numpy array
        Output signal is in microVolts unit.

    """
    resultant_unit = pq.Quantity(1, "uV")  # Final Unit
    for channel in range(len(channel_info)):
        bit_to_volt_conversion = channel_info[channel]["bit_volts"]
        recorded_unit = pq.Quantity([1], channel_info[channel]["units"])
        unit_conversion = (recorded_unit / resultant_unit).simplified
        signal[:, channel] *= bit_to_volt_conversion * unit_conversion
        if "ADC" in channel_info[channel]["channel_name"]:
            signal[:, channel] *= 10**6
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


def load_recording(
    folder: str,
    channel_mask: Optional[Set[int]] = None,
):
    """
    Loads data recorded by Open Ephys in Binary format as numpy memmap.
    The path should contain

    - continuous/<processor name>/continuous.dat: signal (cannot have multiple file)
    - continuous/<processor name>/timestamps.dat: timestamps
    - structure.oebin: number of channels and sampling rate.

    Parameters
    ----------
    folder: str
        folder containing at least the subfolder 'experiment1'.
    channel_mask: Set[int], optional
        Channel index list to ignore in import (default=None)

    Returns
    -------
    signal : SignalType, neo.core.AnalogSignal
    sampling_rate : float

    Raises
    ------
    AssertionError
        If more than one "continuous.dat" file exist in the directory.

    """

    file_path: List[str] = glob(os.path.join(folder, "**", "*.dat"), recursive=True)
    assert (
        len(file_path) == 1
    ), f"There should be only one 'continuous.dat' file. (There exists {file_path})"

    # load structure information dictionary
    info_file: str = os.path.join(folder, "structure.oebin")
    info: Dict[str, Any] = oebin_read(info_file)
    num_channels: int = info["continuous"][0]["num_channels"]
    sampling_rate: float = float(info["continuous"][0]["sample_rate"])
    # channel_info: Dict[str, Any] = info["continuous"][0]["channels"]

    # TODO: maybe need to support multiple continuous.dat files in the future
    signal, timestamps = load_continuous_data(file_path[0], num_channels, sampling_rate)

    # To Voltage
    signal = bits_to_voltage(signal, info["continuous"][0]["channels"])
    # signal = neo.core.AnalogSignal(
    #    signal*pq.uV, sampling_rate=sampling_rate * pq.Hz
    # )

    if channel_mask:
        signal = apply_channel_mask(signal, channel_mask)

    return signal, timestamps, sampling_rate


def load_continuous_data(
    data_path: str,
    num_channels: int,
    sampling_rate: float,
    timestamps_path: Optional[str] = None,
    start_at_zero: bool = True,
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
    start_at_zero : bool
        If True, the timestamps is adjusted to start at zero.
        Note, recorded timestamps might not start at zero for some reason.

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
    raw_data: np.ndarray = np.memmap(data_path, dtype="int16", mode="c")
    length = raw_data.size // num_channels
    raw_data = np.reshape(raw_data, (length, num_channels)).astype("float32")

    # Get timestamps_path
    if timestamps_path is None:
        dirname = os.path.dirname(data_path)
        timestamps_path = os.path.join(dirname, "timestamps.npy")

    # Get timestamps
    if os.path.exists(timestamps_path):
        timestamps = np.array(np.load(timestamps_path), dtype=np.float64)
        timestamps /= float(sampling_rate)
    else:  # If timestamps_path doesn't exist, deduce the stamps
        timestamps = np.array(range(0, length)) / sampling_rate

    # Adjust timestamps to start from zero
    if start_at_zero and not np.isclose(timestamps[0], 0.0):
        timestamps -= timestamps[0]

    return np.array(raw_data), timestamps
