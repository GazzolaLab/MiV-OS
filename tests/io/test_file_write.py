import os
import sys
import time

import h5py as h5
import numpy as np
import pytest

from miv.io import file as miv_file
from tests.io.test_file_read import fixture_mock_h5_file


def isEmpty(dictionary):
    test = True
    print(dictionary.keys())
    for key in dictionary.keys():
        if dictionary[key] is None:
            test = True
        elif type(dictionary[key]) == list or type(dictionary[key]) == np.ndarray:
            if len(dictionary[key]) > 0:
                test = False

    return test


def test_initialize():
    test_data = miv_file.initialize()

    assert isinstance(test_data, dict)


def test_clear_container(mock_h5_file):
    filename = mock_h5_file

    desired_datasets = ["coordinates", "electrodes"]
    subset = 1000

    data, container = miv_file.read(filename, desired_datasets, subset)

    miv_file.clear_container(container)

    assert isEmpty(container)


def test_create_container():
    data = miv_file.initialize()

    miv_file.create_group(data, "coordinates", counter="ncoords")
    miv_file.create_dataset(data, ["px", "py", "pz"], group="coordinates", dtype=float)

    miv_file.create_group(data, "electrodes", counter="nelectrodes")
    miv_file.create_dataset(data, ["u", "v"], group="electrodes", dtype=float)

    test_container = miv_file.create_container(data)

    assert not isEmpty(test_container)
    assert isinstance(test_container, dict)


def test_create_group():
    data = miv_file.initialize()
    miv_file.create_group(data, "coordinates", counter="ncoords")

    assert not isEmpty(data["_GROUPS_"])
    assert "coordinates/ncoords" in data.keys()

    miv_file.create_group(data, "test/slash", counter="ntest/slash")

    assert "test-slash" in data["_GROUPS_"]
    assert "test-slash/ntest-slash" in data.keys()


def test_pack():
    data = miv_file.initialize()
    miv_file.create_group(data, "obj", counter="nobj")
    miv_file.create_dataset(data, ["myfloat"], group="obj", dtype=float)
    miv_file.create_dataset(data, ["myint"], group="obj", dtype=int)
    miv_file.create_dataset(data, ["mystr"], group="obj", dtype=str)

    container = miv_file.create_container(data)

    # Normal packing test

    for i in range(5):
        container["obj/myfloat"].append(2.0)
        container["obj/myint"].append(2)
        container["obj/mystr"].append("two")
    container["obj/nobj"] = 5

    test = miv_file.pack(data, container)
    assert test == 0
    assert len(data["obj/myfloat"]) == 5
    assert len(data["obj/myint"]) == 5
    assert len(data["obj/mystr"]) == 5
    assert data["obj/nobj"][0] == 5

    assert len(container["obj/myfloat"]) == 0
    assert len(container["obj/myint"]) == 0
    assert len(container["obj/mystr"]) == 0
    assert container["obj/nobj"] == 0

    # AUTO_SET_COUNTER = False
    container["obj/myfloat"].append(2.0)
    container["obj/myint"].append(2)
    container["obj/mystr"].append("two")

    # Is the mistake propagated?
    container["obj/nobj"] = 2

    miv_file.pack(data, container, AUTO_SET_COUNTER=False)
    assert data["obj/nobj"][1] == 2

    # Fix mistake
    data["obj/nobj"][1] = 2

    # STRICT_CHECKING = True
    container["obj/myfloat"].append(2.0)
    container["obj/myint"].append(2)
    # 1 != 0, strict checking should fail.

    test = 0
    try:
        miv_file.pack(data, container, STRICT_CHECKING=True)
    except RuntimeError:
        test = -1

    # Was the mistake caught?
    assert test == -1
    # Was nothing packed?
    assert len(data["obj/myint"]) == 6
    # Is container not cleared?
    assert not isEmpty(container)

    # EMPTY_OUT_CONTAINER = False

    container["obj/mystr"].append("two")

    miv_file.pack(data, container, EMPTY_OUT_CONTAINER=False)

    assert not isEmpty(container)

    # assert type(data['obj/mystr'][0]) is str


def test_create_dataset():
    data = miv_file.initialize()
    miv_file.create_group(data, "coordinates", counter="ncoords")
    miv_file.create_dataset(data, ["px", "py", "pz"], group="coordinates", dtype=float)
    miv_file.create_dataset(data, ["e"], group="coordinates", dtype=int)

    assert not isEmpty(data["_GROUPS_"])
    assert "coordinates/ncoords" in data.keys()
    assert "coordinates/px" in data.keys()
    assert "coordinates/e" in data["_MAP_DATASETS_TO_COUNTERS_"].keys()
    assert data["_MAP_DATASETS_TO_COUNTERS_"]["coordinates/e"] == "coordinates/ncoords"
    assert data["_MAP_DATASETS_TO_DATA_TYPES_"]["coordinates/px"] == float
    assert data["_MAP_DATASETS_TO_DATA_TYPES_"]["coordinates/e"] == int


def test_write_metadata(mock_h5_file):
    filename = mock_h5_file
    file = h5.File(filename, "r")

    # Check default attribute existence
    assert "date" in file.attrs.keys()
    # assert 'miv_file_version' in file.attrs.keys()
    assert "h5py_version" in file.attrs.keys()
    assert "numpy_version" in file.attrs.keys()
    assert "python_version" in file.attrs.keys()

    # Check default attributes are strings
    assert isinstance(file.attrs["date"], str)
    # assert isinstance(file.attrs['miv_file_version'], str)
    assert isinstance(file.attrs["h5py_version"], str)
    assert isinstance(file.attrs["numpy_version"], str)
    assert isinstance(file.attrs["python_version"], str)

    file.close()

    # Adding a new attribute
    miv_file.write_metadata(filename, {"author": "John Doe"})
    file = h5.File(filename, "r")

    assert "author" in file.attrs.keys()
    assert file.attrs["author"] == "John Doe"

    file.close()
