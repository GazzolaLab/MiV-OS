__doc__ = """

We expect the data structure to follow the default format exported from OpenEphys system: `format <https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/491632/Data+format>`_.

Original Author

- open-ephys/analysis-tools/Python3/Binary.py (commit: 871e003)
- original author: malfatti
  - date: 2019-07-27
- last modified by: skim449
  - date: 2022-04-11
"""
__all__ = ["Load", "load_data", "load_continuous_data_file"]
from typing import Optional
import numpy as np
from ast import literal_eval
from glob import glob


def ApplyChannelMap(Data, ChannelMap):
    print("Retrieving channels according to ChannelMap... ", end="")
    for R, Rec in Data.items():
        if Rec.shape[1] < len(ChannelMap) or max(ChannelMap) > Rec.shape[1] - 1:
            print("")
            print("Not enough channels in data to apply channel map. Skipping...")
            continue

        Data[R] = Data[R][:, ChannelMap]

    return Data


def BitsToVolts(Data, ChInfo, Unit):
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


def load_data():
    raise NotImplementedError


def load_continuous_data_file(
    data_file: str,
    channels: int,
    timestamps_npy: Optional[str] = "",
    sampling_rate: float = 30000,
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

    raw_data = np.memmap(data_file, dtype="int16")
    length = raw_data.size // channels
    raw_data = np.reshape(raw_data, (length, channels))

    timestamps_zeroed = np.array(range(0, length)) / sampling_rate
    if timestamps_npy == "":
        timestamps = timestamps_zeroed
    else:
        timestamps = np.load(timestamps_npy) / sampling_rate

    # only take first 32 channels
    raw_data = raw_data[:, 0:32]

    return np.array(timestamps), np.array(raw_data)
