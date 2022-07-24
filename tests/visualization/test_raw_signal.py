import os

import numpy as np
import pytest


def test_multi_channel_signal_plot_filecheck(tmp_path):
    from miv.visualization import multi_channel_signal_plot

    signal = np.arange(64 * 128).reshape([128, 64])
    X, Y = np.mgrid[:8, :8]
    mea_geometry = zip(range(64), X.ravel(), Y.ravel())
    output_path = os.path.join(tmp_path, "output.mp4")
    multi_channel_signal_plot(signal, mea_geometry, 0, 128, 10, 30, output_path)
    assert os.path.exists(output_path), "output video file does not exist."
