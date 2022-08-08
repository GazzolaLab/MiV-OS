import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from miv.typing import SignalType


@pytest.fixture
def mock_numpy_signal():
    num_length = 512
    num_channel = 32

    # signal = np.arange(num_length * num_channel).reshape([num_length, num_channel])
    signal_list = []
    x = np.ones(num_length)
    for i in range(num_channel):
        _signal = (i / 1.5) + x + np.random.randn(num_length)  # jitter
        signal_list.append(_signal)
    signal = np.array(signal_list).T
    return signal


@pytest.fixture
def mock_numpy_spiketrains():
    spiketrains = [
        np.sort(np.rint(np.arange(200, 500, 100))).astype(np.int_) for _ in range(32)
    ]
    return spiketrains


@pytest.mark.parametrize("start, end", [(1, 100), (1, 50), (250, 500)])
def test_pairwise_causality_plot_numpy(mock_numpy_signal, start, end):
    from miv.visualization.causality import pairwise_causality_plot

    fig, axes = pairwise_causality_plot(mock_numpy_signal, start, end)

    assert axes.shape == (2, 2), "Dimension of axes does not match."
    assert isinstance(fig, plt.Figure)


@pytest.mark.parametrize("ch1, ch2", [(0, 1), (1, 0)])
@pytest.mark.parametrize("window_length", [10, 20, 50])
def test_spike_triggered_average_plot_numpy(
    mock_numpy_signal, mock_numpy_spiketrains, ch1, ch2, window_length
):
    from miv.visualization.causality import spike_triggered_average_plot

    fig, axes = spike_triggered_average_plot(
        mock_numpy_signal, ch1, mock_numpy_spiketrains, ch2, 1, window_length
    )

    assert isinstance(axes, Axes)
    assert isinstance(fig, plt.Figure)
