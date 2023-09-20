import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from miv.core.datatype import Signal, Spikestamps
from miv.signal.spike import ExtractWaveforms


@pytest.fixture(name="mock_signal_for_waveform")
def fixture_mock_signal_for_waveform_extraction():
    n_length = 100
    signal = Signal(
        data=np.arange(n_length).reshape([n_length, 1]),
        timestamps=np.arange(n_length),
        rate=1,
    )
    return signal


def test_waveform_extraction(mock_signal_for_waveform):
    extract_waveform = ExtractWaveforms(channels=[0], pre=1, post=3)
    extracted_waveform = extract_waveform(mock_signal_for_waveform, Spikestamps([[50]]))
    np.testing.assert_allclose(
        extracted_waveform[0].data, np.array([49, 50, 51, 52])[:, None]
    )


def test_plot_waveform_filecheck(mock_signal_for_waveform, tmp_path):
    extract_waveforms = ExtractWaveforms(channels=[0], pre=1, post=3)
    extracted_waveform = extract_waveforms(
        mock_signal_for_waveform, Spikestamps([[50]])
    )
    extract_waveforms.plot_waveforms(
        extracted_waveform, inputs=None, save_path=tmp_path
    )
    assert os.path.exists(os.path.join(tmp_path, "spike_cutouts_ch000.png"))
