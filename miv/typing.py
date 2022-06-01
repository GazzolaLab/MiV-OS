# from typing import TypeAlias
from typing import Union

import neo
import numpy as np
import numpy.typing as npt

SignalType = Union[
    np.ndarray,
    neo.core.AnalogSignal,  # npt.DTypeLike
]  # Shape should be [signal_length, n_channel]
TimestampsType = np.ndarray
SpikestampsType = Union[np.ndarray, neo.core.SpikeTrain]
