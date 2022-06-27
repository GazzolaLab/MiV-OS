__all__ = ["signal_to_noise"]

import numpy as np

from miv.typing import SignalType


def signal_to_noise(signal: SignalType, axis: int = 0, ddof: int = 0):
    """
    Compute signal-to-noise ratio of raw signal.

    Parameters
    ----------
    signal : SignalType
    axis : int
        Axis of interest. By default, signal axis is 0 (default=1)
    ddof : int
    """
    signal_np = np.asanyarray(signal)
    m = signal_np.mean(axis)
    sd = signal_np.std(axis=axis, ddof=ddof)
    return np.abs(np.where(sd == 0, 0, m / sd))
