import numpy as np
import pytest

"""
def test_binned_spiketrain():
    seg = Segment(index=1)
    train0 = SpikeTrain(
        times=[0.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4, 5, 5.1, 5.2, 8, 9.5],
        units="sec",
        t_stop=10,
    )
    seg.spiketrains.append(train0)
    with np.testing.assert_raises(AssertionError):
        output = binned_spiketrain(seg.spiketrains[0], 0, 0, 0.1)
    # start time must be less than end time
    with np.testing.assert_raises(AssertionError):
        output = binned_spiketrain(seg.spiketrains[0], 0, 5, 0)
    # bin_size cannot be negative
    output = binned_spiketrain(seg.spiketrains[0], 2, 5, 1)
    np.testing.assert_allclose(output, [0, 0, 1])
"""
