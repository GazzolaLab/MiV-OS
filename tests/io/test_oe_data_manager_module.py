import os
import pathlib

import numpy as np
import pytest

from miv.io.openephys import Data, DataManager
from tests.io.mock_data import MockDataGenerator


class MockDataWithFlaw(Data):
    def check_path_validity(self):
        return False


@pytest.fixture(name="mock_data_manager")
def fixture_mock_data_manager_object(tmp_path) -> DataManager:
    n_experimental = 3
    n_recording_per_exp = 5
    branches = ["Record Node 1", "Record Node 2"]

    num_channels = 3
    data_length = 100

    for branch in branches:
        for exp in range(n_experimental):
            for rec in range(n_recording_per_exp):
                path = os.path.join(
                    tmp_path, branch, f"experiment{exp}", f"recording{rec}"
                )
                os.makedirs(path, exist_ok=1)
                MockDataGenerator.create_mock_data_structure(
                    path, num_channels, data_length
                )

    return DataManager(tmp_path)


@pytest.fixture(name="mock_auxillary_data")
def fixture_mock_auxillary_data_object(tmp_path) -> Data:
    path = os.path.join(tmp_path, "Record Node Temp", "experiment0", "recording0")
    os.makedirs(path, exist_ok=1)
    dirname, _, _, _ = MockDataGenerator.create_mock_data_structure(path, 5, 200)
    return Data(dirname)


@pytest.fixture(name="mock_false_data")
def fixture_mock_data_with_path_flaw(tmp_path) -> Data:
    path = os.path.join(tmp_path, "Record Node Temp", "experiment0", "recording0")
    os.makedirs(path, exist_ok=1)
    dirname, _, _, _ = MockDataGenerator.create_mock_data_structure(path, 5, 200)
    return MockDataWithFlaw(dirname)


def test_data_manager_length(mock_data_manager):
    assert len(mock_data_manager) == 30


def test_data_manager_insert(mock_data_manager, mock_auxillary_data):
    mock_data_manager.insert(0, mock_auxillary_data)
    assert len(mock_data_manager) == 31
    mock_data_manager.insert(-1, mock_auxillary_data)
    assert len(mock_data_manager) == 32
    mock_data_manager.insert(1, mock_auxillary_data)
    assert len(mock_data_manager) == 33


def test_data_manager_del(mock_data_manager):
    mock_data_manager.pop()
    assert len(mock_data_manager) == 29
    mock_data_manager.pop()
    assert len(mock_data_manager) == 28


def test_data_manager_extend(mock_data_manager, mock_auxillary_data):
    mock_data_manager.extend([mock_auxillary_data, mock_auxillary_data])
    assert len(mock_data_manager) == 32
    assert mock_data_manager[-1] == mock_auxillary_data
    assert mock_data_manager[-2] == mock_auxillary_data


def test_data_manager_path_list_getter(mock_data_manager):
    result = mock_data_manager.data_path_list
    assert len(result) == 30


def test_data_manager_query_path_by_name(mock_data_manager):
    result = mock_data_manager.query_path_name("experiment2")
    assert len(result) == 10
    result = mock_data_manager.query_path_name("recording1")
    assert len(result) == 6
    result = mock_data_manager.query_path_name("Record Node")
    assert len(result) == 30
    result = mock_data_manager.query_path_name("Record Node 1")
    assert len(result) == 15


def test_data_manager_extend_incorrect_data(mock_data_manager, mock_false_data, caplog):
    mock_data_manager.extend([mock_false_data])
    assert "Invalid data" in caplog.text


def test_data_manager_append_incorrect_data(mock_data_manager, mock_false_data, caplog):
    mock_data_manager.append(mock_false_data)
    assert "Invalid data" in caplog.text


def test_data_manager_insert_incorrect_data(mock_data_manager, mock_false_data, caplog):
    mock_data_manager.insert(0, mock_false_data)
    assert "Invalid data" in caplog.text


def test_data_manager_set_incorrect_data(mock_data_manager, mock_false_data, caplog):
    mock_data_manager.__setitem__(1, mock_false_data)
    assert "Invalid data" in caplog.text
