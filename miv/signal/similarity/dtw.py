__doc__ = """Dynamic time warping modules. """
__all__ = ["dynamic_time_warping_distance_matrix"]
import os
import sys

import numpy as np
from dtaidistance import dtw

from miv.typing import SignalType


def dynamic_time_warping_distance_matrix(
    temporal_sequence: np.ndarray, axis: int = 0, fast: bool = False, **kwargs
):
    """
    Measure dynamic time warping matrix between signals.

    Parameters
    ----------
    temporal_sequence : np.ndarray
        The temporal sequence (time, n_features).
    axis : int
        Axis of time sequence. (default=0)
    fast : bool
        Allow fast computation if `dtaidistance` is compiled with C. (default=False)
        :ref: https://dtaidistance.readthedocs.io/en/latest/usage/installation.html#from-source

    Returns
    -------
    matrix : np.ndarray
        2D matrix with the size (temporal_sequence.shape[axis], temporal_sequence.shape[axis])
        Each element represent the distance between the feature set.
    """
    assert (
        len(temporal_sequence.shape) > 1
    ), f"Sequence must have at least two axes. Provide sequence has a shape {temporal_sequence.shape}."
    if axis != 0:
        temporal_sequence = np.swapaxes(temporal_sequence, axis, 0)

    if fast:
        return dtw.distance_matrix_fast(temporal_sequence, **kwargs)
    else:
        return dtw.distance_matrix(temporal_sequence, **kwargs)
