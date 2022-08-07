import numpy as np
import pytest

from miv.statistics.signal_statistics import signal_to_noise
from miv.typing import SignalType


def test_signal_to_noise():
    signal = np.zeros(10)
    signal[3] = 1.0
    assert signal_to_noise(signal) == 1.0 / 3.0

    signal = np.zeros(10)
    signal[6] = 1.0
    assert signal_to_noise(signal) == 1.0 / 3.0

    signal = np.zeros(10)
    signal[3] = 1.0
    signal[6] = 1.0
    assert np.isclose(signal_to_noise(signal), 1.0 / 2.0)
