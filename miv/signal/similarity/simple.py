__doc__ = """

Similarity Metrics
##################

.. currentmodule:: miv.signal.similarity

.. autofunction:: domain_distance_matrix

"""

__all__ = ["domain_distance_matrix"]
from typing import Literal

import os
import sys

import numba
import numpy as np
from numba import njit

from miv.typing import SignalType

# For beyond python3.11, change to LiteralString

DOMAIN_MOD = Literal["time", "frequency", "power"]


def domain_distance_matrix(temporal_sequence: SignalType, domain: DOMAIN_MOD = "time"):
    """
    Measure similarity between signal using correlation matrix,

    Parameters
    ----------
    temporal_sequence : SignalType
        The temporal sequence (time, n_features).
    domain : Literal['time', 'frequency', 'power']
        If 'time', use time-base correlation to measure the similarity.
        If 'frequency', use FFT-base correlation to measure the similarity.
        If 'power', use power of the signal to measure the similarity.

    Returns
    -------
    matrix : np.ndarray
        2D matrix with the size (n_features, n_features)
        Each element represent the distance between the feature set.
    """
    assert (
        len(temporal_sequence.shape) > 1
    ), f"Sequence must have at least two axes. Provide sequence has a shape {temporal_sequence.shape}."

    n_time, n_features = temporal_sequence.shape
    distance_matrix = np.empty([n_features, n_features], dtype=np.float_)

    if domain == "time":
        func = nb_time_domain_similarity
    elif domain == "frequency":
        func = nb_frequency_domain_similarity
    elif domain == "power":
        func = nb_power_domain_similarity

    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                similarity = func(temporal_sequence[:, i], temporal_sequence[:, j])
                distance_matrix[i, j] = similarity
                distance_matrix[j, i] = similarity

    return distance_matrix


@njit(cache=True)
def nb_time_domain_similarity(signal_a, signal_b):
    reference = np.correlate(signal_a, signal_a)
    corr = np.correlate(signal_a, signal_b)
    return np.abs(reference - corr)


# @njit(cache=True) # TODO: replace fft to other module
def nb_frequency_domain_similarity(signal_a, signal_b):
    reference = np.correlate(np.fft.fft(signal_a), np.fft.fft(signal_a))
    corr = np.correlate(np.fft.fft(signal_a), np.fft.fft(signal_b))
    return np.abs(reference - corr)


@njit(cache=True)
def nb_power_domain_similarity(signal_a, signal_b):
    reference_power = np.sum(np.square(signal_a))
    target_power = np.sum(np.square(signal_b))
    return np.abs(reference_power - target_power)
