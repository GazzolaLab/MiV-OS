import numpy as np
import pytest
import quantities as pq
from neo.core import SpikeTrain

from miv.statistics import firing_rates, peri_stimulus_time

SpikestampsTestSet = [
    [pq.Quantity([1, 2, 3], "s")],
    [pq.Quantity([1, 2, 3], "s"), pq.Quantity([3, 6, 9, 12], pq.s)],
    [SpikeTrain([4, 8, 12], units=pq.s, t_stop=120)],
]
TrueRates = [1, [1, 1.0 / 3], 1.0 / 40]


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
