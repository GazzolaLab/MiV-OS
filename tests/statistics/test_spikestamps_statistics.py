import numpy as np
import pytest
import quantities as pq
from neo.core import SpikeTrain

from miv.statistics import spikestamps_statistics

SpikestampsTestSet = [
    [[1, 2, 3]],
    [[1, 2, 3], [3, 6, 9, 12]],
    [SpikeTrain([4, 8, 12], units=pq.s, t_stop=120)],
]
TrueRates = [1, [1, 1.0 / 3], 1.0 / 40]


@pytest.mark.parametrize("spikestamps, true_rate", zip(SpikestampsTestSet, TrueRates))
def test_spikestamps_statistics_base_function(spikestamps, true_rate):
    result = spikestamps_statistics(spikestamps)
    np.testing.assert_allclose(result["rates"], true_rate)
    assert np.isclose(result["mean"], np.mean(true_rate))
    assert np.isclose(result["variance"], np.var(true_rate))
