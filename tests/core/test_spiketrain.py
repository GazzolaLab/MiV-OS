from quantities import s

from miv.core import SpikeTrain


def test_spiketrain_instantiation():
    assert len(SpikeTrain([1, 2, 3] * s, t_stop=10.0)[2:]) == 1
