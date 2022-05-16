__all__ = ["signal_to_noise"]

import numpy as np

from miv.typing import SignalType

def signal_to_noise(a:SignalType, axis:int=1, ddof:int=0):
    """signal_to_noise.

    Parameters
    ----------
    signal : SignalType
    axis : int
        Axis of interest. By default, channel axis is 1 (default=1)
    ddof : int
    """
    signal_np = np.asanyarray(signal)
    m = signal_np.mean(axis)
    sd = signal_np.std(axis=axis, ddof=ddof)
    return np.where(sd==0, 0, m/sd)
