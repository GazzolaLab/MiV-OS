import os
import time

import h5py as h5
import numpy as np
import pytest

from miv.io import file as miv_file


def isEmpty(dictionary):
    test = True
    print(dictionary.keys())
    for key in dictionary.keys():
        print(key)
        print(dictionary[key])
        print(type(dictionary[key]))
        if dictionary[key] is None:
            test = True
        elif type(dictionary[key]) == list or type(dictionary[key]) == np.ndarray:
            if len(dictionary[key]) > 0:
                test = False

    return test


@pytest.fixture(name="mock_h5_file")
def fixture_mock_h5_file(tmp_path):

    filename = os.path.join(tmp_path, "MiV_TESTS.h5")

    data = miv_file.initialize()

    miv_file.create_group(data, "coordinates", counter="ncoords")
    miv_file.create_dataset(data, ["px", "py", "pz"], group="coordinates", dtype=float)

    miv_file.create_dataset(data, ["u", "v"], group="electrodes", dtype=float)

    event = miv_file.create_container(data)

    for i in range(0, 10):

        ncoords = 5
        event["coordinates/ncoords"] = ncoords

        for n in range(ncoords):
            event["coordinates/px"].append(np.random.random())
            event["coordinates/py"].append(np.random.random())
            event["coordinates/pz"].append(np.random.random())

        event["electrodes/u"].append(np.random.random())
        event["electrodes/v"].append(np.random.random())

        miv_file.pack(data, event)

    miv_file.write(filename, data, comp_type="gzip", comp_opts=9)
    return filename


def test_read(mock_h5_file):
    filename = mock_h5_file

    desired_datasets = ["coordinates", "electrodes"]
    subset = 5

    test_data, test_container = miv_file.read(filename, desired_datasets, subset)

    assert isinstance(test_data, dict)
    assert isinstance(test_container, dict)

    assert isEmpty(test_container)
    assert not isEmpty(test_data)

    # Testing desired_datasets
    assert "coordinates/px" in test_data.keys()

    # Testing subsets
    assert len(test_data["coordinates/ncoords"]) == 5

    test_data, test_container = miv_file.read(filename, desired_datasets, 1000)

    assert len(test_data["coordinates/ncoords"]) == 10

    # Passing in a range of subsets
    subset = (0, 4)
    test_data, test_container = miv_file.read(filename, desired_datasets, subset=subset)
    assert len(test_data["coordinates/ncoords"]) == 4

    subset = (1, 5)
    test_data, test_container = miv_file.read(filename, desired_datasets, subset=subset)
    assert len(test_data["coordinates/ncoords"]) == 4

    subset = [1, 5]
    test_data, test_container = miv_file.read(filename, desired_datasets, subset=subset)
    assert len(test_data["coordinates/ncoords"]) == 4

    # Test for poor uses of subset
    test_data, test_container = miv_file.read(filename, desired_datasets, [0, 0])

    assert len(test_data["_LIST_OF_DATASETS_"]) == 0
    assert len(test_container.keys()) == 0

    test_data, test_container = miv_file.read(filename, desired_datasets, [10, 0])

    assert len(test_data["_LIST_OF_DATASETS_"]) == 0
    assert len(test_container.keys()) == 0

    test_data, test_container = miv_file.read(filename, desired_datasets, subset=0)

    assert len(test_data["_LIST_OF_DATASETS_"]) == 0
    assert len(test_container.keys()) == 0


def test_unpack(mock_h5_file):
    filename = mock_h5_file

    # This assumes you run nosetests from the h5hep directory and not
    # the tests directory.
    desired_datasets = ["coordinates", "electrodes"]
    subset = 10

    container, data = miv_file.read(filename, desired_datasets, subset)

    miv_file.unpack(data, container)

    assert not isEmpty(container)


def test_get_ncontainers_in_file(mock_h5_file):
    filename = mock_h5_file

    ncontainers = miv_file.get_ncontainers_in_file(filename)

    assert ncontainers == 10


def test_get_file_metadata(mock_h5_file):
    filename = mock_h5_file

    metadata = miv_file.get_file_metadata(filename)

    assert "date" in metadata
    assert "h5py_version" in metadata
    assert "numpy_version" in metadata
    assert "python_version" in metadata

    # Check default attributes are strings
    assert isinstance(metadata["date"], str)
    assert isinstance(metadata["h5py_version"], str)
    assert isinstance(metadata["numpy_version"], str)
    assert isinstance(metadata["python_version"], str)
