import numpy as np
import pytest
from neo.core import AnalogSignal, Segment, SpikeTrain

from miv.core.datatype import Spikestamps

# Test set for Fano Factor


# TODO: Finish implementing fano factor for spiketrain
"""
def test_fano_factor_output():
    from miv.statistics import fano_factor

    # Initialize the spiketrain as below
    spikestamps = Spikestamps([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    output = fano_factor(spikestamps, 0, 0, 10, 10)
    # The function above should return two burst events from 1.2 (duration 0.4, length 5, rate 12.5 )
    # and 5(duration 0.2, length 3, rate 15)
    np.testing.assert_allclose(output, 0.0)

def test_fano_factor_empty_spiketrain():
    # Test to throw error in empty spiketrains
    train1 = Spikestamps([])
    with np.testing.assert_raises(AssertionError):
        output = fano_factor(train1, 0, 0, 10, 10)
    # The function above should throw an error since there are no spike to compute variance and mean
    with np.testing.assert_raises(AssertionError):
        output = fano_factor(train1, 0, 0, 0, 10)
    # The function above should throw an error since start time cannot be same or greater than end time
"""
