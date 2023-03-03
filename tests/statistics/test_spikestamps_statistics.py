import numpy as np
import pytest
import quantities as pq
from neo.core import Segment, SpikeTrain

from miv.core.datatype import Spikestamps
from miv.statistics.spiketrain_statistics import (
    firing_rates,
    interspike_intervals,
    peri_stimulus_time,
)

SpikestampsTestSet = [
    Spikestamps([[1, 2, 3]]),
    Spikestamps([[1, 2, 3], [3, 6, 9, 12]]),
    Spikestamps([[4, 8, 12]]),
]
TrueRates = [1.5, [3.0 / 11, 4.0 / 11], 3.0 / 8]
TrueIntervals = [
    [np.array([1, 1])],
    [np.array([1, 1]), np.array([3, 3, 3])],
    [np.array([4, 4])],
]


@pytest.mark.parametrize("spikestamps, true_rate", zip(SpikestampsTestSet, TrueRates))
def test_spikestamps_statistics_base_function(spikestamps, true_rate):
    result = firing_rates(spikestamps)
    np.testing.assert_allclose(result["rates"], true_rate)
    assert np.isclose(result["mean"], np.mean(true_rate))
    assert np.isclose(result["variance"], np.var(true_rate))


SpikeTrainTestSet = [
    [[1, 0, 1, 0], [0, 1, 1, 0]],
    [
        [1, 0, 1, 0],
        [1, 1, 1, 0],
    ],
]
TruePST = [[1, 1, 2, 0], [2, 1, 2, 0]]


@pytest.mark.parametrize("spike_train_set, true_pst", zip(SpikeTrainTestSet, TruePST))
def test_peri_stimulus_time_function(spike_train_set, true_pst):
    PST = peri_stimulus_time(spike_train_set)
    assert np.isclose(PST, true_pst).all()


@pytest.mark.parametrize(
    "spikestamps, true_interval", zip(SpikestampsTestSet, TrueIntervals)
)
def test_interspike_interval_neo(spikestamps, true_interval):
    for spikestamp, interval in zip(spikestamps, true_interval):
        result = interspike_intervals(spikestamp)
        np.testing.assert_allclose(result, interval)
