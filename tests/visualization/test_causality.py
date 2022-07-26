import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from miv.typing import SignalType
from tests.visualization.mock import (
    fixture_mock_numpy_signal,
    fixture_mock_numpy_spiketrains,
)


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
