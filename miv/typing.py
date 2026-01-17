from __future__ import annotations

from typing import TypeAlias, Any
from collections.abc import Callable

import neo
import pathlib
import numpy as np


SignalType = (
    np.ndarray | neo.core.AnalogSignal
)  # Shape should be [signal_length, n_channel]
SpikestampsType = np.ndarray | neo.core.SpikeTrain


PlotCallbackFunc: TypeAlias = Callable[
    [
        Any,  # Outputs
        Any | tuple[Any, ...],  # Inputs
        bool,  # show flag
        pathlib.Path | None,  # save_path
    ],
    None,
]
