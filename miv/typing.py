# from typing import TypeAlias
from typing import Union

import numpy as np
import numpy.typing as npt

import neo

SignalType = Union[
    npt.ArrayLike, np.ndarray, neo.core.AnalogSignal
]  # Shape should be [n_channel, signal_length]
TimestampsType = npt.ArrayLike
SpikestampsType = Union[npt.ArrayLike, neo.core.SpikeTrain]
