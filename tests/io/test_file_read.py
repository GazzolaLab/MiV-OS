import numpy as np
import h5py as h5
import time
from miv.io import file as miv_file
from test_write import test_write_file


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


def test_read():

    test_write_file()

    filename = "MiV_TESTS.h5"
    desired_datasets = ["coordinates", "electrodes"]
    subset = 5

    test_data, test_container = miv_file.read(filename, desired_datasets, subset)

    assert isinstance(test_data, dict)
    assert isinstance(test_container, dict)

    assert isEmpty(test_container) == True
    assert isEmpty(test_data) == False

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


def test_unpack():

    # This assumes you run nosetests from the h5hep directory and not
    # the tests directory.
    filename = "MiV_TESTS.h5"
    desired_datasets = ["coordinates", "electrodes"]
    subset = 10

    container, data = miv_file.read(filename, desired_datasets, subset)

    miv_file.unpack(data, container)

    assert isEmpty(container) == False


def test_get_ncontainers_in_file():

    filename = "MiV_TESTS.h5"

    ncontainers = miv_file.get_ncontainers_in_file(filename)

    assert ncontainers == 10


def test_get_file_metadata():

    filename = "MiV_TESTS.h5"

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
