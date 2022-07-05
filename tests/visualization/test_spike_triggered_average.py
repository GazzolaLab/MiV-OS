import numpy as np
import pytest
from neo.core import AnalogSignal, Segment, SpikeTrain


def test_spike_triggered_avg_output():
    from miv.statistics import burst
    from miv.visualization import spike_triggered_average_plot

    # Initialize the spiketrain as below
    seg = Segment(index=1)
    train0 = SpikeTrain(
        times=[0.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4, 5, 5.1, 5.2, 8, 9.5],
        units="sec",
        t_stop=10,
    )
    signal = np.zeros((100000, 2))
    # train0 = SpikeTrain(times=[0.1,4,5,5.1,5.2,9.5], units='sec', t_stop=10)
    seg.spiketrains.append(train0)

    with np.testing.assert_raises(AssertionError):
        spike_triggered_average_plot(signal, 0, seg.spiketrains, 0, 2000, 10000000)
    # The function above should throw an error since the window is greater than signal length
