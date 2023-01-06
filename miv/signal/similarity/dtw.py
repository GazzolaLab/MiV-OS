__doc__ = """

One could also take additional steps to fine-adjust the shape of the spikes for clustering and classification tasks

Dynamic time warping
####################

Dynamic time warping is a similarity metrics that accounts for the temporal dilatation and shift in multiple sequences of data [1]_ [wiki]_ [dtaidistance]_.

The tooll is available with the sorting extension: `pip install miv-os[sortingExtension]`.

References & Footnotes
======================

.. [1] John Thomasa, Jing Jina, Justin Dauwelsa, Sydney S. Cashb, and M. Brandon Westoverb. (2018), Clustering of Interictal Spikes by Dynamic Time Warping and Affinity Propagation. IEEE Int Conf Acoust Speech Signal Process (2018 March), https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5842698/pdf/nihms946113.pdf
.. [wiki] https://en.wikipedia.org/wiki/Dynamic_time_warping
.. [dtaidistance] https://dtaidistance.readthedocs.io/en/latest/

.. currentmodule:: miv.signal.similarity

.. autofunction:: dynamic_time_warping_distance_matrix

"""
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
