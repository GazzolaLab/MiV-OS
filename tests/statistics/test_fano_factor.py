import numpy as np
import pytest
from neo.core import AnalogSignal, Segment, SpikeTrain

# Test set for Fano Factor


def test_fano_factor_output():
    from miv.statistics import fano_factor

    # Initialize the spiketrain as below
    seg = Segment(index=1)
    train0 = SpikeTrain(
        times=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        units="sec",
        t_stop=10,
    )
    # train0 = SpikeTrain(times=[0.1,4,5,5.1,5.2,9.5], units='sec', t_stop=10)
    seg.spiketrains.append(train0)

    output = fano_factor(seg.spiketrains, 0, 0, 10, 10)
    # The function above should return two burst events from 1.2 (duration 0.4, length 5, rate 12.5 )
    # and 5(duration 0.2, length 3, rate 15)
    np.testing.assert_allclose(output, 0.0)

    # Test to throw error in empty spiketrains
    seg1 = Segment(index=1)
    train1 = SpikeTrain(
        times=[],
        units="sec",
        t_stop=10,
    )
    seg1.spiketrains.append(train1)
    with np.testing.assert_raises(AssertionError):
        output = fano_factor(seg1.spiketrains, 0, 0, 10, 10)
    # The function above should throw an error since there are no spike to compute variance and mean
