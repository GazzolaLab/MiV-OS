import pytest

import numpy as np
from miv.signal.filter import ButterBandpass

from tests.filter.test_filter_protocol import RuntimeFilterProtocol

ParameterSet = [
    (300, 3000, 5, "test set 1"),
    (200, 1000, 3, "test set 2"),
]
ImpossibleParameterSet = [
    (300, 100, 5, "Highcut smaller than lowcut 2"),
    (500, 0, 3, "Highcut smaller than lowcut 2"),
    (500, 2000, -1, "Negative order"),
    (500, 2000, 2.5, "Fractional order"),
]


@pytest.mark.parametrize("lowcut, highcut, order, tag", ParameterSet)
def test_butterworth_filter_protocol_abide(lowcut, highcut, order, tag):
    filt = ButterBandpass(lowcut, highcut, order, tag)
    assert isinstance(filt, RuntimeFilterProtocol)


@pytest.mark.parametrize("lowcut, highcut, order, tag", ImpossibleParameterSet)
def test_butterworth_filter_impossible_initialization(lowcut, highcut, order, tag):
    with pytest.raises(AssertionError):
        ButterBandpass(lowcut, highcut, order, tag)


# fmt: off
AnalyticalTestSet = [  # t: linspace 0->0.005 (w=50)+(w=500)
    (np.array([0., 0.3090169943749475, 0.5877852522924729,
               0.8090169943749478, 0.9510565162951531]),
     30_000,
     np.array([0., 0.00025224672310531133, 0.002500608619913995,
               0.01210053274126219, 0.03851522096001583])),
]
# fmt: on


@pytest.mark.parametrize("lowcut, highcut, order, tag", ParameterSet[:1])
@pytest.mark.parametrize("sig, rate, result", AnalyticalTestSet)
def test_butterworth_filter_analytical(lowcut, highcut, order, tag, sig, rate, result):
    filt = ButterBandpass(lowcut, highcut, order, tag)
    ans = filt(signal=sig, sampling_rate=rate)
    assert np.allclose(ans, result)


@pytest.mark.parametrize("lowcut, highcut, order, tag", ParameterSet)
def test_butterworth_repr_string(lowcut, highcut, order, tag):
    filt = ButterBandpass(lowcut, highcut, order, tag)
    for v in [lowcut, highcut, order, tag]:
        assert str(v) in repr(filt)
