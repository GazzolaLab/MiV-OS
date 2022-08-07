import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from miv.visualization.waveform import extract_waveforms, plot_waveforms


@pytest.fixture(name="mock_signal_for_waveform")
def fixture_mock_signal_for_waveform_extraction():
    n_length = 100
    signal = np.arange(n_length).reshape([n_length, 1])
    return signal


def test_waveform_extraction(mock_signal_for_waveform):
    extracted_waveform = extract_waveforms(
        mock_signal_for_waveform, [[50]], channel=0, sampling_rate=1, pre=1, post=3
    )
    np.testing.assert_allclose(extracted_waveform[0], np.array([49, 50, 51, 52]))


def test_plot_waveform_filecheck(mock_signal_for_waveform, tmp_path):
    extracted_waveform = extract_waveforms(
        mock_signal_for_waveform, [[50]], channel=0, sampling_rate=1, pre=1, post=3
    )
    plot_waveforms(extracted_waveform, 1, pre=1, post=3)
    filename = os.path.join(tmp_path, "savefig.png")
    plt.savefig(filename)
    assert os.path.exists(filename)
