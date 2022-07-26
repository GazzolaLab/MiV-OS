import numpy as np
import pytest

from miv.typing import SignalType, SpikestampsType


@pytest.fixture(name="mock_numpy_signal")
def fixture_mock_numpy_signal() -> SignalType:
    np_rng = np.random.default_rng(
        0
    )  # Some test may not pass. Relation determinant cannot be negative
    num_length = 512
    num_channel = 32

    # signal = np.arange(num_length * num_channel).reshape([num_length, num_channel])
    signal_list = []
    x = np.ones(num_length)
    for i in range(num_channel):
        _signal = (i / 1.7) + (x + np_rng.standard_normal(num_length))  # jitter
        signal_list.append(_signal)
    signal = np.array(signal_list).T
    return signal


@pytest.fixture(name="mock_numpy_spiketrains")
def fixture_mock_numpy_spiketrains():
    spiketrains = [
        np.sort(np.rint(np.arange(200, 500, 100))).astype(np.int_) for _ in range(32)
    ]
    return spiketrains
