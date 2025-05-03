import os
import pathlib

import numpy as np
import pytest

from miv.io.openephys import Data
from tests.io.mock_data import fixture_create_mock_data_file


@pytest.fixture(name="mock_data")
def fixture_mock_data_files(create_mock_data_file) -> Data:
    dirname, _, _, _ = create_mock_data_file
    return Data(dirname)


def test_data_validity(mock_data: Data):
    assert mock_data.check_path_validity()


def test_data_module_readout_files(mock_data: Data):
    mock_data.load()


def test_data_module_data_check(create_mock_data_file):
    (
        dirname,
        expected_signal,
        expected_timestamps,
        expected_sampling_rate,
    ) = create_mock_data_file
    data = Data(dirname)
    for Signal in data.load():
        signal = Signal.data
        timestamps = Signal.timestamps
        sampling_rate = Signal.rate
        np.testing.assert_allclose(signal, expected_signal)
        np.testing.assert_allclose(timestamps, expected_timestamps)
        np.testing.assert_allclose(sampling_rate, expected_sampling_rate)
