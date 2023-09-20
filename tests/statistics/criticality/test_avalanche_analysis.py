from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

import miv.statistics.criticality.avalanche_analysis as module
from miv.core.datatype import Signal, Spikestamps


@patch("%s.module.plt" % __name__)  # mock matplotlib.plt
def test_plot_branching_ratio_histogram(mock_plt, tmp_path):
    # mock plt.subplots returning two values
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)

    # make mock input
    mock_input = np.linspace(0, 1, 100)
    non_zero_input = mock_input[np.nonzero(mock_input)]
    outputs = (None, None, mock_input, None, None)

    ops = module.AvalancheAnalysis()
    ops.plot_branching_ratio_histogram(
        outputs=outputs, inputs=None, show=True, save_path=tmp_path
    )

    assert mock_plt.close.assert_called  # Make sure figures are closed
    assert mock_plt.show.called
    assert mock_plt.savefig.called_with(tmp_path / "branching_ratio.png")
    assert mock_plt.subplots.called

    assert mock_ax.hist.called
    assert mock_ax.hist.called_with(non_zero_input, bins=100)
    assert mock_ax.set_title.called
    assert mock_ax.set_xlabel.called
    assert mock_ax.set_ylabel.called
