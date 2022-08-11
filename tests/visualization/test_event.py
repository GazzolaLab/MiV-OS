import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from miv.typing import SignalType
from tests.visualization.mock import fixture_mock_numpy_spiketrains


@pytest.mark.parametrize("isi", [0.1, 1.1, 0.5, 2.0])
@pytest.mark.parametrize("min_length", [5, 10, 15])
def test_plot_burst(mock_numpy_spiketrains, isi, min_length):
    from miv.visualization.event import plot_burst

    fig, axes = plot_burst(mock_numpy_spiketrains, isi, min_length)

    assert isinstance(axes, Axes)
    assert isinstance(fig, plt.Figure)
