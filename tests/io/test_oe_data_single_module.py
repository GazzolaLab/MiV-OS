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
    for (signal, timestamps, sampling_rate) in data.load(1):
        np.testing.assert_allclose(signal, expected_signal)
        np.testing.assert_allclose(timestamps, expected_timestamps)
        np.testing.assert_allclose(sampling_rate, expected_sampling_rate)


@pytest.mark.parametrize("filename", ["test.png", "test1.jpeg"])
@pytest.mark.parametrize("groupname", ["g1", "_g2"])
@pytest.mark.parametrize(
    "extra_savefig_kwargs", [None, {}, {"dpi": 300, "format": "svg", "pad_inches": 0.2}]
)
def test_data_save_figure_filecheck(
    mock_data, filename, groupname, extra_savefig_kwargs
):
    import matplotlib.pyplot as plt

    fullpath = os.path.join(mock_data.analysis_path, groupname, filename)
    fig = plt.figure()
    mock_data.save_figure(fig, groupname, filename, savefig_kwargs=extra_savefig_kwargs)
    assert os.path.exists(fullpath)
