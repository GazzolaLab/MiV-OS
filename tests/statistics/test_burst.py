import numpy as np
import pytest
from neo.core import AnalogSignal, Segment, SpikeTrain

# Test set For Burst Module


def test_burst_analysis_output():
    from miv.statistics import burst

    # Initialize the spiketrain as below
    seg = Segment(index=1)
    train0 = SpikeTrain(
        times=[0.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4, 5, 5.1, 5.2, 8, 9.5],
        units="sec",
        t_stop=10,
    )
    # train0 = SpikeTrain(times=[0.1,4,5,5.1,5.2,9.5], units='sec', t_stop=10)
    seg.spiketrains.append(train0)

    output = burst(seg.spiketrains, 0, 0.2, 2)
    # The function above should return two burst events from 1.2 (duration 0.4, length 5, rate 12.5 )
    # and 5(duration 0.2, length 3, rate 15)
    np.testing.assert_allclose(output[0], [1.2, 5])
    np.testing.assert_allclose(output[1], [0.4, 0.2])
    np.testing.assert_allclose(output[2], [5, 3])
    np.testing.assert_allclose(output[3], [12.5, 15])

    seg1 = Segment(index=1)
    train1 = SpikeTrain(
        times=[0.1, 1.2, 1.2, 1.2, 5, 6],
        units="sec",
        t_stop=10,
    )
    seg1.spiketrains.append(train1)
    with np.testing.assert_raises(AssertionError):
        output = burst(seg1.spiketrains, 0, 0.2, 2)
    # The function above should throw an error since the rate will become 1/0

    # When the first spike should be in a burst
    seg2 = Segment(index=1)
    train2 = SpikeTrain(
        times=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4, 5, 5.1, 5.2, 8, 9.5],
        units="sec",
        t_stop=10,
    )
    seg2.spiketrains.append(train2)

    output = burst(seg2.spiketrains, 0, 0.2, 2)
    np.testing.assert_allclose(output[0], [1.1, 5])
    np.testing.assert_allclose(output[1], [0.5, 0.2])
    np.testing.assert_allclose(output[2], [6, 3])
    np.testing.assert_allclose(output[3], [12, 15])
