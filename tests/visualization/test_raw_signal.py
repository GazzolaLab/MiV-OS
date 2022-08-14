import os

import numpy as np
import pytest


def test_multi_channel_signal_plot_filecheck(tmp_path):
    from miv.visualization.raw_signal import multi_channel_signal_plot

    length = 64
    channel = 16
    signal = np.arange(channel * length).reshape([length, channel])
    X, Y = np.mgrid[:4, :4]
    mea_geometry = zip(range(16), X.ravel(), Y.ravel())
    output_path = os.path.join(tmp_path, "output.mp4")
    multi_channel_signal_plot(
        signal,
        mea_geometry,
        0,
        length,
        10,
        30,
        output_path,
        max_subplot_in_x=4,
        max_subplot_in_y=4,
    )
    assert os.path.exists(output_path), "output video file does not exist."
