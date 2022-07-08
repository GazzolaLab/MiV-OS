__all__ = [
    "shannon_entropy",
    "block_entropy",
    "entropy_rate",
    "active_information",
    "mutual_information",
    "relative_entropy",
    "conditional_entropy",
    "transfer_entropy",
]


from typing import Any, Dict, Iterable, List, Optional, Union

import datetime

import elephant.statistics
import matplotlib.pyplot as plt
import neo
import numpy as np
import pyinform
import quantities as pq
import scipy
import scipy.signal

from miv.statistics.spiketrain_statistics import binned_spiketrain
from miv.typing import SpikestampsType


def shannon_entropy(
    spiketrains: SpikestampsType,
    channel: float,
    t_start: float,
    t_end: float,
    bin_size: float,
):
    """
    Estimates the shannon entropy for a single channel recording using the binned spiketrain

    Parameters
    ----------
    spiketrains : SpikestampsType
        Single spike-stamps
    channel : float
        electrode/channel
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

    bin_spike = binned_spiketrain(spiketrains, channel, t_start, t_end, bin_size)
    spike_dist = pyinform.dist.Dist(bin_spike)
    shannon_entropy = pyinform.shannon.entropy(spike_dist)
    return shannon_entropy


def block_entropy(
    spiketrains: SpikestampsType,
    channel: float,
    his: float,
    t_start: float,
    t_end: float,
    bin_size: float,
):
    """
    Estimates the block entropy for a single channel recording using the binned spiketrain

    Parameters
    ----------
    spiketrains : SpikestampsType
        Single spike-stamps
    channel : float
        electrode/channel
    his : float
        history length
    t_start : float
        Binning start time
    t_end : float
        Binning end time
    bin_size : float
        bin size in seconds

    Returns
    -------
    block_entropy: float
        Block entropy for the given channel

    """

    bin_spike = binned_spiketrain(spiketrains, channel, t_start, t_end, bin_size)
    block_entropy = pyinform.blockentropy.block_entropy(bin_spike, his)
    return block_entropy


def entropy_rate(
    spiketrains: SpikestampsType,
    channel: float,
    his: float,
    t_start: float,
    t_end: float,
    bin_size: float,
):
    """
    Estimates the entropy rate for a single channel recording using the binned spiketrain

    Parameters
    ----------
    spiketrains : SpikestampsType
        Single spike-stamps
    channel : float
        electrode/channel
    his : float
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

    bin_spike = binned_spiketrain(spiketrains, channel, t_start, t_end, bin_size)
    entropy_rate = pyinform.entropyrate.entropy_rate(bin_spike, k=his)
    return entropy_rate


def active_information(
    spiketrains: SpikestampsType,
    channel: float,
    his: float,
    t_start: float,
    t_end: float,
    bin_size: float,
):
    """
    Estimates the active information for a single channel recording using the binned spiketrain

    Parameters
    ----------
    spiketrains : SpikestampsType
        Single spike-stamps
    channel : float
        electrode/channel
    his : float
        history length
    t_start : float
        Binning start time
    t_end : float
        Binning end time
    bin_size : float
        bin size in seconds

    Returns
    -------
    active_information: float
        Active information for the given channel

    """

    bin_spike = binned_spiketrain(spiketrains, channel, t_start, t_end, bin_size)
    active_information = pyinform.activeinfo.active_info(bin_spike, his)
    return active_information


def mutual_information(
    spiketrains: SpikestampsType,
    channelx: float,
    channely: float,
    t_start: float,
    t_end: float,
    bin_size: float,
):
    """
    Estimates the mutual information for the pair of electorde recordings (X & Y) using the binned spiketrains

    Parameters
    ----------
    spiketrains : SpikestampsType
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

    bin_spike_x = binned_spiketrain(spiketrains, channelx, t_start, t_end, bin_size)
    bin_spike_y = binned_spiketrain(spiketrains, channely, t_start, t_end, bin_size)
    mutual_information = pyinform.mutualinfo.mutual_info(bin_spike_x, bin_spike_y)
    return mutual_information


def relative_entropy(
    spiketrains: SpikestampsType,
    channelx: float,
    channely: float,
    t_start: float,
    t_end: float,
    bin_size: float,
):
    """
    Estimates the relative entropy for the pair of electorde recordings (X & Y) using the binned spiketrains

    Parameters
    ----------
    spiketrains : SpikestampsType
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

    bin_spike_x = binned_spiketrain(spiketrains, channelx, t_start, t_end, bin_size)
    bin_spike_y = binned_spiketrain(spiketrains, channely, t_start, t_end, bin_size)
    relative_entropy = pyinform.relativeentropy.relative_entropy(
        bin_spike_x, bin_spike_y
    )
    return relative_entropy


def conditional_entropy(
    spiketrains: SpikestampsType,
    channelx: float,
    channely: float,
    t_start: float,
    t_end: float,
    bin_size: float,
):
    """
    Estimates the conditional entropy for the pair of electorde recordings (X & Y) using the binned spiketrains

    Parameters
    ----------
    spiketrains : SpikestampsType
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

    bin_spike_x = binned_spiketrain(spiketrains, channelx, t_start, t_end, bin_size)
    bin_spike_y = binned_spiketrain(spiketrains, channely, t_start, t_end, bin_size)
    conditional_entropy = pyinform.conditionalentropy.conditional_entropy(
        bin_spike_x, bin_spike_y
    )
    return conditional_entropy


def transfer_entropy(
    spiketrains: SpikestampsType,
    channelx: float,
    channely: float,
    his: float,
    t_start: float,
    t_end: float,
    bin_size: float,
):
    """
    Estimates the transfer entropy for the pair of electorde recordings (X & Y) using the binned spiketrains and history

    Parameters
    ----------
    spiketrains : SpikestampsType
        Single spike-stamps
    channelx : float
        electrode/channel X
    channely : float
        electrode/channel Y
    his : float
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

    bin_spike_x = binned_spiketrain(spiketrains, channelx, t_start, t_end, bin_size)
    bin_spike_y = binned_spiketrain(spiketrains, channely, t_start, t_end, bin_size)
    transfer_entropy = pyinform.transferentropy.transfer_entropy(
        bin_spike_x, bin_spike_y, k=his
    )
    return transfer_entropy
