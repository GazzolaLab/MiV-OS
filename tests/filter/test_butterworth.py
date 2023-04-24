import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.signal as sps

from miv.core.datatype import Signal
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


AnalyticalTestSet = [  # t: linspace 0->0.005 (w=50)+(w=500)
    (
        Signal(
            data=np.arange(50).reshape(50, 1) * 0.01,
            timestamps=np.linspace(0, 0.005, 50),
            rate=30000,
        ),
        np.array(
            [
                [
                    0.0158288,
                    0.01546371,
                    0.01502895,
                    0.0145279,
                    0.01396378,
                    0.01334069,
                    0.01266394,
                    0.01193998,
                    0.01117588,
                    0.01037881,
                    0.00955546,
                    0.00871186,
                    0.00785333,
                    0.00698465,
                    0.00611034,
                    0.00523486,
                    0.00436276,
                    0.00349863,
                    0.0026471,
                    0.00181266,
                    0.00099957,
                    0.00021177,
                    -0.00054715,
                    -0.00127394,
                    -0.00196562,
                    -0.00261951,
                    -0.00323312,
                    -0.0038042,
                    -0.00433068,
                    -0.00481075,
                    -0.00524279,
                    -0.00562543,
                    -0.00595757,
                    -0.0062384,
                    -0.00646742,
                    -0.00664449,
                    -0.00676982,
                    -0.00684397,
                    -0.00686777,
                    -0.00684227,
                    -0.00676861,
                    -0.00664802,
                    -0.00648176,
                    -0.00627128,
                    -0.00601837,
                    -0.00572533,
                    -0.00539517,
                    -0.00503153,
                    -0.00463856,
                    -0.00422052,
                ]
            ]
        ).T,
    ),
]


@pytest.mark.parametrize("lowcut, highcut, order, tag", ParameterSet[:1])
@pytest.mark.parametrize("sig, result", AnalyticalTestSet)
def test_butterworth_filter_analytical(lowcut, highcut, order, tag, sig, result):
    filt = ButterBandpass(lowcut, highcut, order, tag)
    ans = filt(signal=sig)
    np.testing.assert_allclose(ans.data, result, rtol=1e-5)


@pytest.mark.parametrize("lowcut, highcut, order, tag", ParameterSet)
def test_butterworth_repr_string(lowcut, highcut, order, tag):
    filt = ButterBandpass(lowcut, highcut, order, tag)
    for v in [lowcut, highcut, order, tag]:
        assert str(v) in repr(filt)


def test_butterworth_call_invalid_signal_shape():
    # Create a ButterBandpass object with default parameters
    filt = ButterBandpass(lowcut=5, highcut=10)
    # Create a signal with an invalid shape (more than 2 dimensions)
    with pytest.raises(AssertionError):
        filt(
            Signal(
                data=np.random.randn(10, 10, 10),
                timestamps=np.linspace(0, 1, 10),
                rate=10000,
            )
        )


# TODO: The function plot_frequency_response is disabled.
"""
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
        fig.axes[0].get_title() == "Butterworth filter (order5) frequency response [5,10]"
    )
"""


def test_butter_bandpass_highpass():
    # Create a ButterBandpass object with btype set to "highpass"
    filter = ButterBandpass(lowcut=5, highcut=10, btype="highpass")

    # Set the sampling rate
    sampling_rate = 1000

    # Call the _butter_bandpass method
    sos = filter._butter_bandpass(sampling_rate)
    targ = np.array(
        [
            [0.95043584, -0.95043584, 0.0, 1.0, -0.96906742, 0.0],
            [1.0, -2.0, 1.0, 1.0, -1.94947342, 0.95043584],
            [1.0, -2.0, 1.0, 1.0, -1.97979631, 0.9807737],
        ]
    )

    # Check that the returned filter coefficients are correct
    np.testing.assert_allclose(sos, targ, rtol=1e-5)


def test_butter_bandpass_lowpass():
    # Create a ButterBandpass object with btype set to "lowpass"
    filter = ButterBandpass(lowcut=5, highcut=10, btype="lowpass")

    # Set the sampling rate
    sampling_rate = 1000

    # Call the _butter_bandpass method
    sos = filter._butter_bandpass(sampling_rate)
    targ = np.array(
        [
            [
                2.76887141e-08,
                5.53774282e-08,
                2.76887141e-08,
                1.00000000e00,
                -9.39062506e-01,
                0.00000000e00,
            ],
            [
                1.00000000e00,
                2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.89955855e00,
                9.03314303e-01,
            ],
            [
                1.00000000e00,
                1.00000000e00,
                0.00000000e00,
                1.00000000e00,
                -1.95806055e00,
                9.61931972e-01,
            ],
        ]
    )

    # Check that the returned filter coefficients are correct
    np.testing.assert_allclose(sos, targ, rtol=1e-5)
