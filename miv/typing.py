# from typing import TypeAlias
from typing import Union

import numpy as np
import numpy.typing as npt

import neo

SignalType = Union[
    np.ndarray,
    neo.core.AnalogSignal,  # npt.DTypeLike
]  # Shape should be [n_channel, signal_length]
TimestampsType = np.ndarray
SpikestampsType = Union[np.ndarray, neo.core.SpikeTrain]
