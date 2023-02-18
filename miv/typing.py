# from typing import TypeAlias
from typing import List, Union

import neo
import numpy as np
import numpy.typing as npt

import miv
import miv.core
import miv.core.datatype

SignalType = Union[
    np.ndarray, neo.core.AnalogSignal  # npt.DTypeLike
]  # Shape should be [signal_length, n_channel]
TimestampsType = np.ndarray
SpikestampsType = Union[np.ndarray, neo.core.SpikeTrain, miv.core.datatype.Spikestamps]
SpiketrainType = np.ndarray  # non-sparse boolean

DataTypes = Union[SignalType, SpikestampsType]
