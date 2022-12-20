from typing import Any, Dict, List, Optional, Sequence, Type, Union

import datetime
import sys
from logging import Logger

import h5py
import numpy as np
from h5py._hl.files import File
from numpy import bytes_


def initialize() -> Dict[str, Any]:
    """Creates an empty data dictionary

    Returns:

        **data** (dict): An empty data dictionary

    """

    data: Dict[str, Any] = {}

    data["_GROUPS_"] = {}
    data["_MAP_DATASETS_TO_COUNTERS_"] = {}
    data["_LIST_OF_COUNTERS_"] = []
    data["_MAP_DATASETS_TO_DATA_TYPES_"] = {}
    data["_GROUP_METADATA_"] = {}

    data["_PROTECTED_NAMES_"] = [
        "_PROTECTED_NAMES_",
        "_GROUPS_",
        "_MAP_DATASETS_TO_COUNTERS_",
        "_MAP_DATASETS_TO_DATA_TYPES_",
        "_LIST_OF_COUNTERS_",
        "_GROUP_METADATA_",
    ]

    return data


def clear_container(container: Dict[str, Any]) -> None:
    """Clears the data from the container dictionary.

    Args:
        **container** (dict): The container to be cleared.

    """

    for key in container.keys():

        if key == "_LIST_OF_COUNTERS_":
            continue

        if isinstance(container[key], list):
            container[key].clear()
        elif isinstance(container[key], np.ndarray):
            container[key] = []
        elif isinstance(container[key], int):
            if key in container["_LIST_OF_COUNTERS_"]:
                container[key] = 0
            else:
                container[key] = -999
        elif isinstance(container[key], float):
            container[key] = np.nan
        elif isinstance(container[key], str):
            container[key] = ""


def create_container(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Creates a container dictionary that will be used to collect data and then
    packed into the the master data dictionary.

    Args:
        **data** (dict): Data dictionary that will hold all the data from the container.

    Returns:
        **container** (dict): The new container dictionary with keys and no container information

    """

    container: Dict[str, Any] = {}

    for k in data.keys():
        if k in data["_LIST_OF_COUNTERS_"]:
            container[k] = 0
        else:
            container[k] = data[k].copy()

    return container


def create_group(
    data: Dict[str, Any],
    group_name: str,
    metadata: Dict[str, Union[str, int, float]] = {},
    counter: Optional[str] = None,
    logger: Optional[Logger] = None,
) -> str:
    """Adds a group in the dictionary

    Args:
        **data** (dict): Dictionary to which the group will be added

        **group_name** (string): Name of the group to be added

        **counter** (string): Name of the counter key. None by default

    """

    group_id = group_name.replace("/", "-")
    if logger is not None:
        logger.warning(
            "----------------------------------------------------"
            f"Slashes / are not allowed in group names"
            f"Replacing / with - in group name {group_name}"
            f"The new name will be {group_id}"
            "----------------------------------------------------"
        )
    # Change name of variable, just to keep code more understandable
    counter_id = counter

    # Create a counter_name if the user has not specified one
    if counter_id is None:
        counter_id = f"N_{group_id}"
        if logger is not None:
            logger.warning(
                "----------------------------------------------------"
                f"There is no counter to go with group {group_name}"
                f"Creating a counter called {counter_id}"
                "-----------------------------------------------------"
            )

    # Check for slashes in the counter name. We can't have them.
    if counter_id.find("/") >= 0:
        counter_id = counter_id.replace("/", "-")

    keys = data.keys()

    # Then put the group and any datasets in there next.
    keyfound = False
    for k in keys:
        if group_id == k:
            if logger is not None:
                logger.warning(f"{group_name} is already in the dictionary!")
            keyfound = True
            break

    if not keyfound:

        data["_GROUPS_"][group_id] = []

        data["_GROUPS_"][group_id].append(counter_id)
        counter_path = f"{group_id}/{counter_id}"

        data["_GROUP_METADATA_"][group_id] = metadata
        data["_MAP_DATASETS_TO_COUNTERS_"][group_id] = counter_path
        data["_MAP_DATASETS_TO_DATA_TYPES_"][counter_path] = int

        if counter_path not in data["_LIST_OF_COUNTERS_"]:
            data["_LIST_OF_COUNTERS_"].append(counter_path)

        data[counter_path] = []

    return group_id


def create_dataset(
    data: Dict[str, Any],
    datasets: Union[str, List[str]],
    group: str,
    dtype: Union[Type[int], Type[float], Type[str]] = float,
    logger: Optional[Logger] = None,
) -> int:
    """Adds a dataset to a group in a dictionary. If the group does not exist, it will be created.

    Args:
        **data** (dict): Dictionary that contains the group

        **datasets** (list): Datasets to be added to the group

        **group** (string): Name of group the dataset will be added to.

        **dtype** (type): The data type. float by default.

    Returns:
        **-1**: If the group is None


    """

    if isinstance(datasets, str):
        datasets_: List[str] = [datasets]
    else:
        datasets_ = datasets

    # Check for slashes in the group name. We can't have them.
    for i in range(len(datasets_)):
        dataset_name = datasets_[i]
        if dataset_name.find("/") >= 0:
            new_dataset_name = dataset_name.replace("/", "-")
            datasets_[i] = new_dataset_name
            if logger is not None:
                logger.warning(
                    "----------------------------------------------------"
                    f"Slashes / are not allowed in dataset names"
                    f"Replacing / with - in dataset name {dataset_name}"
                    f"The new name will be {new_dataset_name}"
                    "----------------------------------------------------"
                )

    keys = data.keys()

    if group.find("/") >= 0:
        group = group.replace("/", "-")

    # Put the counter in the dictionary first.
    keyfound = False
    for k in data["_GROUPS_"]:
        if group == k:
            keyfound = True

    if not keyfound:
        counter = f"N_{group}"
        create_group(data, group, counter=counter)
        if logger is not None:
            logger.warning(
                f"Group {group} is not in the dictionary yet!"
                f"Adding it, along with a counter of {counter}"
            )

    # Then put the datasets into the group in there next.
    for dataset in datasets_:
        keyfound = False
        name = f"{group}/{dataset}"
        for k in keys:
            if name == k:
                if logger is not None:
                    logger.warning(f"{name} is already in the dictionary!")
                keyfound = True
        if not keyfound:
            if logger is not None:
                logger.info(
                    f"Adding dataset {dataset} to the dictionary under group {group}."
                )
            data[name] = []
            data["_GROUPS_"][group].append(dataset)

            # Add a counter for this dataset for the group with which it is associated.
            counter = data["_MAP_DATASETS_TO_COUNTERS_"][group]
            # counter_name = "%s/%s" % (group,counter)
            data["_MAP_DATASETS_TO_COUNTERS_"][name] = counter

            data["_MAP_DATASETS_TO_DATA_TYPES_"][name] = dtype

    return 0


def pack(
    data: Dict[str, Any],
    container: Dict[str, Any],
    AUTO_SET_COUNTER: bool = True,
    EMPTY_OUT_CONTAINER: bool = True,
    STRICT_CHECKING: bool = False,
    logger: Optional[Logger] = None,
) -> int:
    """Takes the data from an container and packs it into the data dictionary,
    intelligently, so that it can be stored and extracted efficiently.

    Args:
        **data** (dict): Data dictionary to hold the entire dataset EDIT.

        **container** (dict): container to be packed into data.

        **EMPTY_OUT_CONTAINER** (bool): If this is `True` (default) then empty out the container in preparation
                                for the next iteration. Useful to disable when debugging and inspecting containers.

        **STRICT_CHECKING** (bool): If `True`, then check that all datasets have the same length, otherwise

        **AUTO_SET_COUNTER** (bool): If `True`, update counter value with length of dataset in container

    """

    # Calculate the number of entries for each group and set the
    # value of that counter.
    if AUTO_SET_COUNTER:
        for group in data["_GROUPS_"]:

            datasets = data["_GROUPS_"][group]
            counter = data["_MAP_DATASETS_TO_COUNTERS_"][group]

            # Here we will calculate the values for the counters, based
            # on the size of the datasets
            counter_value = None

            # Loop over the datasets
            for d in datasets:
                full_dataset_name = group + "/" + d
                # Skip any counters
                if counter == full_dataset_name:
                    continue
                else:
                    # Grab the size of the first dataset
                    temp_counter_value = len(container[full_dataset_name])

                    # If we're not STRICT_CHECKING, then use that value for the
                    # counter and break the loop over the datasets, moving on
                    # to the next group.
                    if STRICT_CHECKING is False:
                        container[counter] = temp_counter_value
                        break
                    # Otherwise, we'll check that *all* the datasets have the same
                    # length.
                    else:
                        if counter_value is None:
                            counter_value = temp_counter_value
                            container[counter] = temp_counter_value
                        elif counter_value != temp_counter_value:
                            # In this case, we found two groups of different length!
                            # Print this to help the user identify their error
                            if logger is not None:
                                logger.warning(
                                    f"Two datasets in group {group} have different sizes!"
                                )
                            for tempd in datasets:
                                temp_full_dataset_name = group + "/" + tempd
                                # Don't worry about the dataset
                                if counter == temp_full_dataset_name:
                                    continue

                            # Return a value for the external program to catch.
                            raise RuntimeError(
                                f"Two datasets in group {group} have different sizes!"
                            )

    # Then pack the container into the data
    keys = list(container.keys())
    for key in keys:

        if key in data["_PROTECTED_NAMES_"]:
            continue

        if isinstance(container[key], list):
            value = container[key]
            if len(value) > 0:
                data[key] += value
        else:
            data[key].append(container[key])

    # Clear out the container after it has been packed
    if EMPTY_OUT_CONTAINER:
        clear_container(container)

    return 0


def convert_list_and_key_to_string_data(datalist, key):
    """Converts data dictionary to a string

    Args:
        **datalist** (list): A list to be saved as a string.

    Returns:
        **key** (string): We will assume that this will be unpacked as a dictionary,
                      and this will be the key for the list in that dictionary.

    """

    a = np.string_(key)

    mydataset = []
    b = np.string_("")
    nvals = len(datalist)
    for i, val in enumerate(datalist):
        b += np.string_(val)
        if i < nvals - 1:
            b += np.string_("__:__")
    mydataset.append([a, b])

    return mydataset


def convert_dict_to_string_data(dictionary: Dict[str, str]) -> List[List[bytes_]]:
    """Converts data dictionary to a string

    Args:
        **dictionary** (dict): Dictionary to be converted to a string

    Returns:
        **mydataset** (string): String representation of the dataset

    """

    keys = dictionary.keys()

    mydataset = []
    for i, key in enumerate(keys):
        a = np.string_(key)
        b = np.string_(dictionary[key])
        mydataset.append([a, b])

    return mydataset


def write_metadata(
    filename: str,
    metadata: Dict[str, str] = {},
    write_default_values: bool = True,
    append: bool = True,
) -> File:

    """Writes file metadata in the attributes of an HDF5 file

    Args:
    **filename** (string): Name of output file

    **metadata** (dictionary): Metadata desired by user

    **write_default_values** (boolean): True if user wants to write/update the
                                        default metadata: date, hepfile version,
                                        h5py version, numpy version, and Python
                                        version, false if otherwise.

    **append** (boolean): True if user wants to keep older metadata, false otherwise.

    Returns:
    **hdoutfile** (HDF5): File with new metadata

    """

    hdoutfile = h5py.File(filename, "a")

    non_metadata = ["_NUMBER_OF_CONTAINERS_", "_NUMBER_OF_ENTRIES_"]

    if not append:
        for key in hdoutfile.attr.keys():
            if key not in non_metadata:
                del hdoutfile.attrs[key]

    if write_default_values:
        hdoutfile.attrs["date"] = datetime.datetime.now().isoformat(sep=" ")
        hdoutfile.attrs["numpy_version"] = np.__version__
        hdoutfile.attrs["h5py_version"] = h5py.__version__
        hdoutfile.attrs["python_version"] = sys.version

    for key in metadata:
        hdoutfile.attrs[key] = metadata[key]

    hdoutfile.close()
    return hdoutfile


def write(
    filename: str,
    data: Dict[str, Any],
    comp_type: Optional[str] = None,
    comp_opts: Optional[int] = None,
    logger: Optional[Logger] = None,
) -> File:

    """Writes the selected data to an HDF5 file

    Args:
        **filename** (string): Name of output file

        **data** (dictionary): Data to be written into output file

        **comp_type** (string): Type of compression

    Returns:
        **hdoutfile** (HDF5): File to which the data has been written

    """
    hdoutfile = h5py.File(filename, "w", libver="latest", rdcc_nbytes=1024**3)

    _GROUPS_ = data["_GROUPS_"].keys()

    # Convert this to a 2xN array for writing to the hdf5 file.
    # This gives us one small list of information if we need to pull out
    # small chunks of data
    mydataset = convert_dict_to_string_data(data["_MAP_DATASETS_TO_COUNTERS_"])
    dset = hdoutfile.create_dataset(
        "_MAP_DATASETS_TO_COUNTERS_",
        data=mydataset,
        dtype="S256",
        compression=comp_type,
        compression_opts=comp_opts,
    )

    # Convert this to a 2xN array for writing to the hdf5 file.
    # This has the _GROUPS_ and the datasets in them.
    for group in _GROUPS_:

        hdoutfile.create_group(group)
        hdoutfile[group].attrs["counter"] = np.string_(
            data["_MAP_DATASETS_TO_COUNTERS_"][group]
        )

        metadata = data["_GROUP_METADATA_"][group]
        for key in metadata:
            val = metadata[key]
            if isinstance(val, str):
                hval = np.string_(val)
            else:
                hval = val
            hdoutfile[group].attrs[key] = hval

        datasets = data["_GROUPS_"][group]

        for dataset in datasets:

            name = None
            name = f"{group}/{dataset}"

            x = data[name]

            dataset_dtype = data["_MAP_DATASETS_TO_DATA_TYPES_"][name]

            if isinstance(x, list):
                x = np.asarray(x, dtype=dataset_dtype)

            if logger is not None:
                logger.info(
                    f"Writing dataset {name} to file {name} as type {str(dataset_dtype)}: x.dtype  {x.dtype} data.shape = {x.shape}"
                )

            dset = None
            if dataset_dtype is not str:
                dset = hdoutfile.create_dataset(
                    name,
                    data=x,
                    compression=comp_type,
                    compression_opts=comp_opts,
                    dtype=dataset_dtype,
                    chunks=True,
                )
            else:
                # For writing strings, ensure strings are ascii and not Unicode
                dataset_dtype = h5py.special_dtype(vlen=str)
                longest_word = len(max(x, key=len))
                arr = np.array(x, dtype="S" + str(longest_word))
                dset = hdoutfile.create_dataset(
                    name,
                    data=arr,
                    dtype=dataset_dtype,
                    compression=comp_type,
                    compression_opts=comp_opts,
                )
            dset.attrs["_GROUP_"] = np.string_(group)

    # Get the number of containers
    counters = data["_LIST_OF_COUNTERS_"]
    _NUMBER_OF_CONTAINERS_ = -1
    prevcounter = None
    for i, countername in enumerate(counters):
        ncounter = len(data[countername])
        if logger is not None:
            logger.debug(f"{countername:<32s} has {ncounter:<12d} entries")
        if i > 0 and ncounter != _NUMBER_OF_CONTAINERS_:
            if logger is not None:
                logger.warning(
                    f"{countername} and {prevcounter} have differing numbers of entries!"
                )

        if _NUMBER_OF_CONTAINERS_ < ncounter:
            _NUMBER_OF_CONTAINERS_ = ncounter

        prevcounter = countername

    hdoutfile.attrs["_NUMBER_OF_CONTAINERS_"] = _NUMBER_OF_CONTAINERS_
    hdoutfile.close()

    write_metadata(filename)

    return hdoutfile
