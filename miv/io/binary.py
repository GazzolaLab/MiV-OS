__doc__ = """

-------------------------------------

We expect the data structure to follow the default format exported from OpenEphys system: `format <https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/491632/Data+format>`_.

Original Author

- open-ephys/analysis-tools/Python3/Binary.py (commit: 871e003)
- original author: malfatti
  - date: 2019-07-27
- last modified by: skim449
  - date: 2022-04-11

Raw Data Loader
###############

"""
__all__ = ["load_continuous_data", "load_recording", "oebin_read", "apply_channel_mask"]

from typing import Any, Dict, Optional, Union, List

import os
import numpy as np
from ast import literal_eval
from glob import glob
import quantities as pq
import neo

from miv.typing import SignalType, TimestampsType


def apply_channel_mask(signal: np.ndarray, channel_mask: List[int]):
    """Apply channel mask on the given signal.

    Parameters
    ----------
    signal : np.ndarray
        Shape of the signal is expected to be (num_data_point, num_channels).
    channel_mask : List[int]

    Returns
    -------
    output signal : SignalType

    """
    print("Retrieving channels according to channel_mask... ", end="")
    for R, Rec in signal.items():
        if Rec.shape[1] < len(channel_mask) or max(channel_mask) > Rec.shape[1] - 1:
            print("")
            print("Not enough channels in data to apply channel map. Skipping...")
            continue

        signal[R] = signal[R][:, channel_mask]

    return Data


    signal, timestamps = load_continuous_data(file_path, num_channels, sampling_rate)
    if channel_mask is not None:
        signal = apply_channel_mask(signal, channel_mask)

    # TODO in the future: check inside the channel_info,
    #       and convert mismatch unit (mV->uV)

    signal = neo.core.AnalogSignal(signal.T, unit=unit, sampling_rate=sampling_rate)
    return signal, timestamps, sampling_rate


def _bitsToVolts(Data, ChInfo, Unit):  # TODO: need refactor
    print("Converting to uV... ", end="")
    Data = {R: Rec.astype("float32") for R, Rec in Data.items()}

    if Unit.lower() == "uv":
        U = 1
    elif Unit.lower() == "mv":
        U = 10 ** -3

    for R in Data.keys():
        for C in range(len(ChInfo)):
            Data[R][:, C] = Data[R][:, C] * ChInfo[C]["bit_volts"] * U
            if "ADC" in ChInfo[C]["channel_name"]:
                Data[R][:, C] *= 10 ** 6

    return Data


def Load(
    Folder, Processor=None, Experiment=None, Recording=None, Unit="uV", ChannelMap=[]
):
    """
    Loads data recorded by Open Ephys in Binary format as numpy memmap.

        Load(Folder, Processor=None, Experiment=None, Recording=None, Unit='uV', ChannelMap=[])

    Parameters
    ----------
        Folder: str
            Folder containing at least the subfolder 'experiment1'.

        Processor: str or None, optional
            Processor number to load, according to subsubsubfolders under
            Folder>experimentX/recordingY/continuous . The number used is the one
            after the processor name. For example, to load data from the folder
            'Channel_Map-109_100.0' the value used should be '109'.
            If not set, load all processors.

        Experiment: int or None, optional
            Experiment number to load, according to subfolders under Folder.
            If not set, load all experiments.

        Recording: int or None, optional
            Recording number to load, according to subsubfolders under Folder>experimentX .
            If not set, load all recordings.

        Unit: str or None, optional
            Unit to return the data, either 'uV' or 'mV' (case insensitive). In
            both cases, return data in float32. Defaults to 'uV'.
            If anything else, return data in int16.

        ChannelMap: list, optional
            If empty (default), load all channels.
            If not empty, return only channels in ChannelMap, in the provided order.
            CHANNELS ARE COUNTED STARTING AT 0.

    Returns
    -------
        Data: dict
            Dictionary with data in the structure Data[Processor][Experiment][Recording].

        Rate: dict
            Dictionary with sampling rates in the structure Rate[Processor][Experiment].


    Example
    -------
        import Binary

        Folder = '/home/user/PathToData/2019-07-27_00-00-00'
        Data, Rate = Binary.Load(Folder)

        ChannelMap = [0,15,1,14]
        Recording = 3
        Data2, Rate2 = Binary.Load(Folder, Recording=Recording, ChannelMap=ChannelMap, Unit='Bits')
    """

    Files = sorted(glob(Folder + "/**/*.dat", recursive=True))
    InfoFiles = sorted(glob(Folder + "/*/*/structure.oebin"))

    Data, Rate = {}, {}
    for F, File in enumerate(Files):
        File = File.replace("\\", "/")  # Replace windows file delims
        Exp, Rec, _, Proc = File.split("/")[-5:-1]
        Exp = str(int(Exp[10:]) - 1)
        Rec = str(int(Rec[9:]) - 1)
        Proc = Proc.split(".")[0].split("-")[-1]
        if "_" in Proc:
            Proc = Proc.split("_")[0]

        if Proc not in Data.keys():
            Data[Proc], Rate[Proc] = {}, {}

        if Experiment:
            if int(Exp) != Experiment - 1:
                continue

        if Recording:
            if int(Rec) != Recording - 1:
                continue

        if Processor:
            if Proc != Processor:
                continue

        print("Loading recording", int(Rec) + 1, "...")
        if Exp not in Data[Proc]:
            Data[Proc][Exp] = {}
        Data[Proc][Exp][Rec] = np.memmap(File, dtype="int16", mode="c")

        Info = literal_eval(open(InfoFiles[F]).read())
        ProcIndex = [
            Info["continuous"].index(_)
            for _ in Info["continuous"]
            if str(_["source_processor_id"]) == Proc
        ][
            0
        ]  # Changed to source_processor_id from recorded_processor_id

        ChNo = Info["continuous"][ProcIndex]["num_channels"]
        if Data[Proc][Exp][Rec].shape[0] % ChNo:
            print("Rec", Rec, "is broken")
            del Data[Proc][Exp][Rec]
            continue

        SamplesPerCh = Data[Proc][Exp][Rec].shape[0] // ChNo
        Data[Proc][Exp][Rec] = Data[Proc][Exp][Rec].reshape((SamplesPerCh, ChNo))
        Rate[Proc][Exp] = Info["continuous"][ProcIndex]["sample_rate"]

    for Proc in Data.keys():
        for Exp in Data[Proc].keys():
            if Unit.lower() in ["uv", "mv"]:
                ChInfo = Info["continuous"][ProcIndex]["channels"]
                Data[Proc][Exp] = BitsToVolts(Data[Proc][Exp], ChInfo, Unit)

            if ChannelMap:
                Data[Proc][Exp] = ApplyChannelMap(Data[Proc][Exp], ChannelMap)

    print("Done.")

    return (Data, Rate)


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
    timestamps: TimestampsType, numpy array
    raw_data: SignalType, numpy array

    Raises
    ------
    FileNotFoundError
        If data_path is invalid.
    ValueError
        If the error message shows the array cannot be reshaped due to shape,
        make sure the num_channels is set accurately.

    """

    # Read raw data signal
    raw_data: np.ndarray = np.memmap(data_path, dtype="int16")
    length = raw_data.size // num_channels
    raw_data = np.reshape(raw_data, (length, num_channels))

    # Get timestamps_path
    if timestamps_path is None:
        dirname = os.path.dirname(data_path)
        timestamps_path = os.path.join(dirname, "timestamps.npy")

    # Get timestamps
    if os.path.exists(timestamps_path):
        timestamps = np.load(timestamps_path)
        timestamps /= sampling_rate
    else:  # If timestamps_path doesn't exist, deduce the stamps
        timestamps = np.array(range(0, length)) / sampling_rate

    # Adjust timestamps to start from zero
    if start_at_zero and not np.isclose(timestamps[0], 0.0):
        timestamps -= timestamps[0]

    return timestamps, raw_data
