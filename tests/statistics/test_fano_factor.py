import numpy as np
import pytest
from neo.core import AnalogSignal, Segment, SpikeTrain

from miv.core import Spikestamps
from miv.statistics import fano_factor

# Test set for Fano Factor


def test_fano_factor_output():
    # Initialize the spiketrain as below
    spikestamps = Spikestamps([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    output = fano_factor(spikestamps)
    np.testing.assert_allclose(output, 0.998)


def test_fano_factor_empty_spiketrain():
    # Test to throw error in empty spiketrains
    train1 = Spikestamps([])
    with pytest.raises(AssertionError):
        fano_factor(train1, t_start=10, t_end=1)
    # The function above should throw an error since there are no spike to compute variance and mean
    with pytest.raises(AssertionError):
        fano_factor(train1, -1)
    # The function above should throw an error since start time cannot be same or greater than end time
