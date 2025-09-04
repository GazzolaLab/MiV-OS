import numpy as np
import pytest

from miv.core import Signal, Spikestamps
from miv.statistics.peristimulus_analysis import PSTH, peri_stimulus_time

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


def test_psth(tmp_path):
    # create example events and spikestamps
    events = Spikestamps([np.array([0.01, 0.03, 0.05]), np.array([])])
    spikestamps = Spikestamps(
        [
            np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.035]),
            np.array([0.01, 0.03, 0.05]),
        ]
    )

    # create PSTH instance
    psth = PSTH(binsize=0.001, interval=0.1)
    psth.set_save_path(tmp_path)

    # call PSTH with example events and spikestamps
    result = psth(events, spikestamps)

    # check that result is a Signal object with correct shape
    assert isinstance(result, Signal)
    assert result.data.shape == (100, 2)

    # check that result timestamps and rate are correct
    assert np.allclose(result.timestamps, np.linspace(0, 0.1, 100) + 0.01)
    assert np.isclose(result.rate, 1000.0)
