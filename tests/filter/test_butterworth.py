import matplotlib.pyplot as plt
import numpy as np
import pytest

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


def test_butterworth_call_invalid_signal_shape():
    # Create a ButterBandpass object with default parameters
    filter = ButterBandpass(lowcut=5, highcut=10)

    # Create a signal with an invalid shape (more than 2 dimensions)
    signal = np.random.randn(10, 10, 10)

    # Set the sampling rate
    sampling_rate = 1000

    # Check that calling the filter with the invalid signal shape raises a ValueError
    with pytest.raises(ValueError):
        filter(signal, sampling_rate)


def test_plot_frequency_response():
    # Create a ButterBandpass object with default parameters
    filter = ButterBandpass(lowcut=5, highcut=10)

    # Set the sampling rate
    sampling_rate = 1000

    # Call the plot_frequency_response method
    fig = filter.plot_frequency_response(sampling_rate)

    # Check that the figure object is returned
    assert isinstance(fig, plt.Figure)

    # Check that the figure has the expected title
    assert (
        fig.axes[0].get_title()
        == "Butterworth filter (order5) frequency response [5,10]"
    )


def test_butter_bandpass_highpass():
    # Create a ButterBandpass object with btype set to "highpass"
    filter = ButterBandpass(lowcut=5, highcut=10, btype="highpass")

    # Set the sampling rate
    sampling_rate = 1000

    # Call the _butter_bandpass method
    b, a = filter._butter_bandpass(sampling_rate)

    # Check that the returned filter coefficients are correct
    assert b == pytest.approx(
        [0.95043584, -4.7521792, 9.5043584, -9.5043584, 4.7521792, -0.95043584]
    )
    assert a == pytest.approx(
        [1.0, -4.89833715, 9.59849709, -9.40530799, 4.60847636, -0.90332829]
    )


def test_butter_bandpass_lowpass():
    # Create a ButterBandpass object with btype set to "lowpass"
    filter = ButterBandpass(lowcut=5, highcut=10, btype="lowpass")

    # Set the sampling rate
    sampling_rate = 1000

    # Call the _butter_bandpass method
    b, a = filter._butter_bandpass(sampling_rate)

    # Check that the returned filter coefficients are correct
    assert b == pytest.approx(
        [
            2.76887141e-08,
            1.38443571e-07,
            2.76887141e-07,
            2.76887141e-07,
            1.38443571e-07,
            2.76887141e-08,
        ]
    )
    assert a == pytest.approx(
        [1.0, -4.7966816, 9.20724238, -8.84036968, 4.24578647, -0.81597668]
    )
