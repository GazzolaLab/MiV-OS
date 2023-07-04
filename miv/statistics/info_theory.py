__all__ = [
    "probability_distribution",
    "shannon_entropy",
    "block_entropy",
    "entropy_rate",
    "active_information",
    "mutual_information",
    "relative_entropy",
    "joint_entropy",
    "conditional_entropy",
    "cross_entropy",
    "transfer_entropy",
    "partial_information_decomposition",
]


from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Union

import datetime

import elephant.statistics
import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq
import scipy
import scipy.signal
from tqdm import tqdm

from miv.core.datatype import Spikestamps
from miv.statistics.spiketrain_statistics import binned_spiketrain

INFO_METRICS_TAG = Literal["self", "pair", "all"]  # Pragma: no cover


def tag_info_metrics(tag: INFO_METRICS_TAG) -> Callable:  # Pragma: no cover
    """
    Decorator to tag the info metrics functions
        - self: channel-wise metrics
        - pair: channel pair-wise metrics
        - all: combined across all channels
    """

    def decorator(func: Callable) -> Callable:
        func.tag = tag
        return func

    return decorator


@tag_info_metrics("self")
def probability_distribution(
    spiketrains: Spikestamps,
    bin_size: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
):

    """
    Forms the probability distribution required to compute the information theory measures. Probability is computed based on the binned spiketrain generated for the specified bin size.

    Parameters
    ----------
    spiketrains : Spikestamps
    bin_size : float
        bin size in seconds
    t_start : Optional[float]
        Binning start time. If None, the start time is the first spike time.
    t_end : Optional[float]
        Binning end time. If None, the end time is the last spike time.

    Returns
    -------
        probability_distribution: np.ndarray
        probability distribution for the provided spiketrain

    """
    bin_spike = spiketrains.binning(bin_size=bin_size, t_start=t_start, t_end=t_end)
    prob_spike = (
        bin_spike.data.sum(axis=bin_spike._SIGNALAXIS, keepdims=True)
        / bin_spike.shape[bin_spike._SIGNALAXIS]
    )
    probability_distribution = np.empty_like(bin_spike.data, dtype=np.float_)
    _prob_spike = np.repeat(
        prob_spike, bin_spike.shape[bin_spike._SIGNALAXIS], axis=bin_spike._SIGNALAXIS
    )
    probability_distribution[bin_spike.data] = _prob_spike[bin_spike.data]
    probability_distribution[~bin_spike.data] = (1 - _prob_spike)[~bin_spike.data]
    return probability_distribution


@tag_info_metrics("self")
def shannon_entropy(
    spiketrains: Spikestamps,
    bin_size: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
):
    """
    Estimates the shannon entropy for a single channel recording using the binned spiketrain.

    Parameters
    ----------
    spiketrains : Spikestamps
        Single spike-stamps
    t_start : float
        Binning start time
    t_end : float
        Binning end time
    bin_size : float
        bin size in seconds

    Returns
    -------
        shannon_entropy: float
        Shannon entropy for the given channel

    """
    bin_spike = spiketrains.binning(bin_size=bin_size, t_start=t_start, t_end=t_end)
    prob_spike = (
        bin_spike.data.sum(axis=bin_spike._SIGNALAXIS, keepdims=True)
        / bin_spike.shape[bin_spike._SIGNALAXIS]
    ).astype(np.float_)
    prob_no_spike = 1 - prob_spike
    shannon_entropy = -(
        prob_spike * np.log2(prob_spike) + prob_no_spike * np.log2(prob_no_spike)
    )
    return shannon_entropy


@tag_info_metrics("self")
def block_entropy(
    spiketrains: Spikestamps,
    history: int,
    bin_size: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
):
    """
    Estimates the block entropy for a recording using the binned spiketrain

    Parameters
    ----------
    spiketrains : Spikestamps
        Single spike-stamps
    history : int
        history length
    bin_size : float
        bin size in seconds
    t_start : float
        Binning start time
    t_end : float
        Binning end time

    Returns
    -------
    block_entropy: float
        Block entropy for the given channel

    """
    import pyinform

    assert history > 0, "history length should be a finite positive value"
    bin_spike = spiketrains.binning(bin_size=bin_size, t_start=t_start, t_end=t_end)
    block_entropy = np.zeros((bin_spike.shape[bin_spike._CHANNELAXIS],))
    for idx, bin_spike_channel in enumerate(bin_spike):
        block_entropy[idx] = pyinform.blockentropy.block_entropy(
            bin_spike_channel, history
        )
    return block_entropy


@tag_info_metrics("self")
def entropy_rate(
    spiketrains: Spikestamps,
    history: float,
    bin_size: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
):
    """
    Estimates the entropy rate for a each channel recording using the binned spiketrain

    Parameters
    ----------
    spiketrains : Spikestamps
        Single spike-stamps
    history : int
        history length
    t_start : float
        Binning start time
    t_end : float
        Binning end time
    bin_size : float
        bin size in seconds

    Returns
    -------
    entropy rate: float
        entropy rate for the given channel

    """
    import pyinform

    assert history > 0, "history length should be a finite positive value"
    bin_spike = spiketrains.binning(bin_size=bin_size, t_start=t_start, t_end=t_end)
    entropy_rate = []
    for idx, bin_spike_channel in enumerate(bin_spike):
        entropy_rate.append(
            pyinform.entropyrate.entropy_rate(bin_spike_channel, k=history, local=True)[
                0
            ]
        )
    return np.asarray(entropy_rate), bin_spike  # TODO: return single type


@tag_info_metrics("self")
def active_information(
    spiketrains: Spikestamps,
    history: float,
    bin_size: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
):
    """
    Estimates the active information for a single channel recording using the binned spiketrain

    Parameters
    ----------
    spiketrains : Spikestamps
        Single spike-stamps
    bin_size : float
        bin size in seconds
    history : float
        history length
    t_start : float
        Binning start time
    t_end : float
        Binning end time

    Returns
    -------
    active_information: float
        Active information for the given channel

    """
    import pyinform

    assert history > 0, "history length should be a finite positive value"
    bin_spike = spiketrains.binning(bin_size=bin_size, t_start=t_start, t_end=t_end)
    active_information = np.zeros((bin_spike.shape[bin_spike._CHANNELAXIS],))
    for idx, bin_spike_channel in enumerate(bin_spike):
        active_information[idx] = pyinform.activeinfo.active_info(
            bin_spike_channel, k=history
        )
    return active_information


@tag_info_metrics("pair")
def transfer_entropy(
    spiketrains: Spikestamps,
    history: float,
    bin_size: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
):
    """
    Estimates the transfer entropy for the pair of electorde recordings (X & Y) using the binned spiketrains and history

    Parameters
    ----------
    spiketrains : Spikestamps
        Single spike-stamps
    history : float
        history length
    t_start : float
        Binning start time
    t_end : float
        Binning end time
    bin_size : float
        bin size in seconds

    Returns
    -------
    transfer_entropy: float
        Transfer_entropy for the given pair of electrodes

    """
    import pyinform

    assert history > 0, "history length should be a finite positive value"
    bin_spike = spiketrains.binning(bin_size=bin_size, t_start=t_start, t_end=t_end)

    transfer_entropy_matrix = np.zeros(
        [bin_spike.number_of_channels, bin_spike.number_of_channels]
    )
    for channelx in tqdm(range(bin_spike.number_of_channels)):
        for channely in tqdm(range(bin_spike.number_of_channels), leave=False):
            bin_spike_x = bin_spike[channelx]
            bin_spike_y = bin_spike[channely]

            # np.random.shuffle(bin_spike_x)

            normalizer = pyinform.entropyrate.entropy_rate(
                bin_spike_y, k=history, local=False
            )
            # normalizer = pyinform.mutualinfo.mutual_info(bin_spike_x, bin_spike_y, local=False)

            transfer_entropy = pyinform.transferentropy.transfer_entropy(
                bin_spike_x, bin_spike_y, k=history, local=False
            )
            if np.isclose(normalizer, 0):
                transfer_entropy_matrix[channelx, channely] = 0
            else:
                transfer_entropy_matrix[channelx, channely] = (
                    transfer_entropy / normalizer
                )
    return transfer_entropy_matrix


@tag_info_metrics("pair")
def mutual_information(
    spiketrains: Spikestamps,
    channelx: float,
    channely: float,
    bin_size: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
):
    """
    Estimates the mutual information for the pair of electorde recordings (X & Y) using the binned spiketrains

    Parameters
    ----------
    spiketrains : Spikestamps
        Single spike-stamps
    channelx : float
        electrode/channel X
    channely : float
        electrode/channel Y
    t_start : float
        Binning start time
    t_end : float
        Binning end time
    bin_size : float
        bin size in seconds

    Returns
    -------
    mutual_information: float
        Mutual information for the given pair of electrodes

    """
    import pyinform

    assert t_start < t_end, "start time cannot be equal or greater than end time"
    assert bin_size > 0, "bin_size should be a finite positive value"
    bin_spike_x = binned_spiketrain(spiketrains, channelx, t_start, t_end, bin_size)
    bin_spike_y = binned_spiketrain(spiketrains, channely, t_start, t_end, bin_size)
    mutual_information = pyinform.mutualinfo.mutual_info(bin_spike_x, bin_spike_y)
    return mutual_information


@tag_info_metrics("pair")
def joint_entropy(
    spiketrains: Spikestamps,
    channelx: float,
    channely: float,
    bin_size: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
):
    """
    Estimates the joint entropy for the pair of electorde recordings (X & Y) using the binned spiketrains

    Parameters
    ----------
    spiketrains : Spikestamps
        Single spike-stamps
    channelx : float
        electrode/channel X
    channely : float
        electrode/channel Y
    t_start : float
        Binning start time
    t_end : float
        Binning end time
    bin_size : float
        bin size in seconds

    Returns
    -------
    joint_entropy: float
        joint entropy for the given pair of electrodes

    """
    assert t_start < t_end, "start time cannot be equal or greater than end time"
    assert bin_size > 0, "bin_size should be a finite positive value"

    spike_dist_x = probability_distribution(
        spiketrains, channelx, t_start, t_end, bin_size
    )
    spike_dist_y = probability_distribution(
        spiketrains, channely, t_start, t_end, bin_size
    )
    spike_dist_xy = np.logical_and(spike_dist_x, spike_dist_y)
    joint_entropy = -np.sum(spike_dist_xy * np.log2(spike_dist_xy))
    return joint_entropy


@tag_info_metrics("pair")
def relative_entropy(
    spiketrains: Spikestamps,
    channelx: float,
    channely: float,
    bin_size: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
):
    """
    Estimates the relative entropy for the pair of electorde recordings (X & Y) using the binned spiketrains

    Parameters
    ----------
    spiketrains : Spikestamps
        Single spike-stamps
    channelx : float
        electrode/channel X
    channely : float
        electrode/channel Y
    t_start : float
        Binning start time
    t_end : float
        Binning end time
    bin_size : float
        bin size in seconds

    Returns
    -------
    relative_entropy: float
        Relative_entropy for the given pair of electrodes

    """
    import pyinform

    assert t_start < t_end, "start time cannot be equal or greater than end time"
    assert bin_size > 0, "bin_size should be a finite positive value"

    bin_spike_x = binned_spiketrain(spiketrains, channelx, t_start, t_end, bin_size)
    bin_spike_y = binned_spiketrain(spiketrains, channely, t_start, t_end, bin_size)
    relative_entropy = pyinform.relativeentropy.relative_entropy(
        bin_spike_x, bin_spike_y
    )
    return relative_entropy


@tag_info_metrics("pair")
def conditional_entropy(
    spiketrains: Spikestamps,
    channelx: float,
    channely: float,
    bin_size: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
):
    """
    Estimates the conditional entropy for the pair of electorde recordings (X & Y) using the binned spiketrains

    Parameters
    ----------
    spiketrains : Spikestamps
        Single spike-stamps
    channelx : float
        electrode/channel X
    channely : float
        electrode/channel Y
    t_start : float
        Binning start time
    t_end : float
        Binning end time
    bin_size : float
        bin size in seconds

    Returns
    -------
    conditional_entropy: float
        conditional entropy for the given pair of electrodes

    """
    import pyinform

    assert t_start < t_end, "start time cannot be equal or greater than end time"
    assert bin_size > 0, "bin_size should be a finite positive value"

    bin_spike_x = binned_spiketrain(spiketrains, channelx, t_start, t_end, bin_size)
    bin_spike_y = binned_spiketrain(spiketrains, channely, t_start, t_end, bin_size)
    conditional_entropy = pyinform.conditionalentropy.conditional_entropy(
        bin_spike_x, bin_spike_y
    )
    return conditional_entropy


@tag_info_metrics("pair")
def cross_entropy(
    spiketrains: Spikestamps,
    channelx: float,
    channely: float,
    bin_size: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
):
    """
    Estimates the cross entropy for the pair of electorde recordings (X & Y) using the binned spiketrains

    Parameters
    ----------
    spiketrains : Spikestamps
        Single spike-stamps
    channelx : float
        electrode/channel X
    channely : float
        electrode/channel Y
    t_start : float
        Binning start time
    t_end : float
        Binning end time
    bin_size : float
        bin size in seconds

    Returns
    -------
    cross_entropy: float
        cross entropy for the given pair of electrodes

    """
    assert t_start < t_end, "start time cannot be equal or greater than end time"
    assert bin_size > 0, "bin_size should be a finite positive value"

    spike_dist_x = probability_distribution(
        spiketrains, channelx, t_start, t_end, bin_size
    )
    spike_dist_y = probability_distribution(
        spiketrains, channely, t_start, t_end, bin_size
    )
    cross_entropy = -np.sum(spike_dist_x * np.log2(spike_dist_y))
    return cross_entropy


@tag_info_metrics("pair")
def partial_information_decomposition(
    spiketrains: Spikestamps,
    channelx: float,
    channely: float,
    channelz: float,
    bin_size: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
):
    """
    Decomposes the information provided by channel x and y about channel z in redundancy, unique information, and synergy.

    Parameters
    ----------
    spiketrains : Spikestamps
        Single spike-stamps
    channelx : float
        electrode/channel X
    channely : float
        electrode/channel Y
    channelz : float
        electrode/channel Y
    t_start : float
        Binning start time
    t_end : float
        Binning end time
    bin_size : float
        bin size in seconds

    Returns
    -------
    redundancy: float
        redundant information provided by both x and y about z
    unique_information_x: float
        information uniquely provided by x about z
    unique_information_y: float
        information uniquely provided by y about z
    synergy: float
        synergetic information provided by both x and y about z
    total_information: float
        total information provided by both x and y about z
    """
    assert t_start < t_end, "start time cannot be equal or greater than end time"
    assert bin_size > 0, "bin_size should be a finite positive value"

    spike_dist_x = probability_distribution(
        spiketrains, channelx, t_start, t_end, bin_size
    )
    spike_dist_y = probability_distribution(
        spiketrains, channely, t_start, t_end, bin_size
    )
    spike_dist_z = probability_distribution(
        spiketrains, channely, t_start, t_end, bin_size
    )
    spike_dist_xz = np.logical_and(spike_dist_x, spike_dist_z)
    spike_dist_yz = np.logical_and(spike_dist_y, spike_dist_z)
    spike_dist_xy = np.logical_and(spike_dist_x, spike_dist_y)
    spike_dist_xyz = np.logical_and(spike_dist_x, spike_dist_y, spike_dist_z)
    I_x_z = np.sum(
        spike_dist_xz * np.log2(spike_dist_xz / (spike_dist_x * spike_dist_z))
    )
    I_y_z = np.sum(
        spike_dist_yz * np.log2(spike_dist_yz / (spike_dist_y * spike_dist_z))
    )
    redundancy = np.min(I_x_z, I_y_z)
    unique_information_x = I_x_z - redundancy
    unique_information_y = I_y_z - redundancy
    total_information = np.sum(
        spike_dist_xyz * np.log2(spike_dist_xyz / (spike_dist_xy * spike_dist_z))
    )
    synergy = (
        total_information - redundancy - unique_information_x - unique_information_y
    )
    return (
        redundancy,
        unique_information_x,
        unique_information_y,
        synergy,
        total_information,
    )
