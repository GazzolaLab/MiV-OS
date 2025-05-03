# from typing import TypeAlias

import neo
import numpy as np


SignalType = (
    np.ndarray | neo.core.AnalogSignal
)  # Shape should be [signal_length, n_channel]
SpikestampsType = np.ndarray | neo.core.SpikeTrain
