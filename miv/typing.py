from typing import TypeAlias
from typing import Union, Tuples

import numpy as np
import numpy.typing as npt

import neo

SignalType: TypeAlias = Union[npt.ArrayLike, np.ndarray, neo.core.AnalogSignal]
TimestampsType: TypeAlias = npt.ArrayLike
