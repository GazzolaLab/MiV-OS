from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

import miv.statistics.criticality.avalanche_analysis as module
from miv.core.datatype import Signal, Spikestamps


@patch("%s.module.plt" % __name__)  # mock matplotlib.plt
def test_avalanche_plot(mock_plt):
    # mock plt.subplots returning two values
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    mock_input = Spikestamps([[1, 5, 10]])
    mock_signal = Signal(
        data=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        timestamps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        rate=1,
    )

    ops = module.AvalancheDetection()
    ops.receive = MagicMock(return_value=[mock_input])  # Mock "receive" method
    ops.plot_avalanche_on_raster(
        outputs=[np.array([0, 2]), np.array([1, 3]), mock_signal],
        inputs=None,
    )
    ops.receive.assert_called_once_with()

    assert mock_plt.close.assert_called
    assert mock_plt.subplots.called
