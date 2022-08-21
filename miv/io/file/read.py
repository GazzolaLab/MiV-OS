from typing import Any, Dict, List, Optional, Tuple, Union

from logging import Logger

import h5py
import numpy as np
from h5py._hl.dataset import Dataset
from numpy import int64, ndarray


def read(
    filename: str,
    groups: Optional[Union[str, List[str]]] = None,
    subset: Optional[Union[int, List[int], Tuple[int, int]]] = None,
    logger: Optional[Logger] = None,
) -> Tuple[Dict[str, Any], Dict[str, None]]:
    """
    Reads all, or a subset of the data, from the HDF5 file to fill a data dictionary.
    Returns an empty dictionary to be filled later with data from individual containers.

    Args:
        **filename** (string): Name of the input file

        **groups** (list): Dataset groups to be read from input file.

        **subset** (int): Number of containers to be read from input file

        **logger** (logging.Logger): optional logger object

    Returns:
        **data** (dict): Selected data from HDF5 file

        **container** (dict): An empty container dictionary to be filled by data from select containers

    """

    # Open the HDF5 file
    infile = None
    infile = h5py.File(filename, "r")

    # Create the initial data and container dictionary to hold the data
    data: Dict[str, Any] = {}
    container: Dict[str, Any] = {}

    data["_MAP_DATASETS_TO_COUNTERS_"] = {}
    data["_MAP_DATASETS_TO_INDEX_"] = {}
    data["_LIST_OF_COUNTERS_"] = []
    data["_LIST_OF_DATASETS_"] = []
    data["_GROUPS_"] = []
    data["_MAP_DATASETS_TO_GROUPS_"] = {}

    # Get the number of containers
    data["_NUMBER_OF_CONTAINERS_"] = infile.attrs["_NUMBER_OF_CONTAINERS_"]

    ncontainers = data["_NUMBER_OF_CONTAINERS_"]

    # Determine if only a subset of the data should be read
    subset_: Union[None, List[int]] = None
    if subset is not None:

        try:
            subset_ = validate_subset(subset, ncontainers)
        except RuntimeError:
            infile.close()
            return data, container

        data["_NUMBER_OF_CONTAINERS_"] = subset_[1] - subset_[0]
        ncontainers = data["_NUMBER_OF_CONTAINERS_"]

        if logger is not None:
            logger.info(
                "Will read in a subset of the file!"
                f"From container {subset_[0]} (inclusive) through container {subset_[1]-1} (inclusive)"
                f"Container {subset_[1]} is not read in"
                f"Reading in {ncontainers} containers\n"
            )

    # Get the datasets and counters
    dc = infile["_MAP_DATASETS_TO_COUNTERS_"]
    for vals in dc:

        # The decode is there because vals were stored as numpy.bytes
        counter = vals[1].decode()
        index_name = f"{counter}_INDEX"
        data["_MAP_DATASETS_TO_COUNTERS_"][vals[0].decode()] = counter
        data["_MAP_DATASETS_TO_INDEX_"][vals[0].decode()] = index_name
        data["_LIST_OF_COUNTERS_"].append(vals[1].decode())
        data["_LIST_OF_DATASETS_"].append(vals[0].decode())
        data["_LIST_OF_DATASETS_"].append(vals[1].decode())  # Get the counters as well

    # We may have added some counters and datasets multiple times.
    # So just to be sure, only keep the unique values
    data["_LIST_OF_COUNTERS_"] = np.unique(data["_LIST_OF_COUNTERS_"]).tolist()
    data["_LIST_OF_DATASETS_"] = np.unique(data["_LIST_OF_DATASETS_"]).tolist()

    # Get the list of datasets and groups
    all_datasets = data["_LIST_OF_DATASETS_"]

    all_datasets = select_datasets(all_datasets, groups, logger=logger)

    # Pull out the counters and build the indices
    # We will need to keep track of the indices in the entire file
    # This way, if the user specifies a subset of the data, we have the full
    # indices already calculated
    full_file_indices = {}

    for counter_name in data["_LIST_OF_COUNTERS_"]:

        full_file_counters = infile[counter_name]
        full_file_index = calculate_index_from_counters(full_file_counters)

        # If we passed in subset, grab that slice of the data from the file
        if subset_ is not None:
            # Add 1 to the high range of subset when we pull out the counters
            # and index because in order to get all of the entries for the last entry.
            data[counter_name] = infile[counter_name][subset_[0] : subset_[1] + 1]
            index: np.ndarray = full_file_index[subset_[0] : subset_[1] + 1]
        else:
            data[counter_name] = infile[counter_name][:]
            index = full_file_index

        subset_index = index
        # If the file is *not* empty....
        # Just to make sure the "local" index of the data dictionary starts at 0
        if len(index) > 0:
            subset_index = index - index[0]

        index_name = f"{counter_name}_INDEX"

        data[index_name] = subset_index
        full_file_indices[index_name] = index

    # Loop over the all_datasets we want and pull out the data.
    for name in all_datasets:

        # If this is a counter, we're going to have to grab the indices
        # differently than for a "normal" dataset
        IS_COUNTER = True
        index_name_: Union[None, str] = None
        if name not in data["_LIST_OF_COUNTERS_"]:
            index_name_ = data["_MAP_DATASETS_TO_INDEX_"][name]
            IS_COUNTER = False  # We will use different indices for the counters

        dataset = infile[name]

        # This will ignore the groups
        if isinstance(dataset, h5py.Dataset):
            dataset_name = name
            group_name = dataset.attrs.get("_GROUP_", None)

            if subset_ is not None:
                if IS_COUNTER:
                    # If this is a counter, then the subset indices
                    # map on to the same locations for any counters
                    lo = subset_[0]
                    hi = subset_[1]
                else:
                    if index_name_ is not None:
                        lo = full_file_indices[index_name_][0]
                        hi = full_file_indices[index_name_][-1]
                    else:
                        raise RuntimeError("Unknown index")
                data[dataset_name] = dataset[lo:hi]
            else:
                data[dataset_name] = dataset[:]
            if group_name not in data["_GROUPS_"]:
                data["_GROUPS_"].append(group_name)
            if dataset_name not in data["_MAP_DATASETS_TO_GROUPS_"]:
                data["_MAP_DATASETS_TO_GROUPS_"][dataset_name] = group_name

            container[
                dataset_name
            ] = None  # This will be filled for individual container

    infile.close()
    return data, container


def select_datasets(
    datasets: List[str],
    groups: Optional[Union[str, List[str]]] = None,
    logger: Optional[Logger] = None,
) -> List[str]:

    # Only keep select data from file, if we have specified datasets
    if groups is not None:

        if isinstance(groups, str):
            groups = list(groups)

        # Count backwards because we'll be removing stuff as we go.
        i = len(datasets) - 1
        while i >= 0:
            entry = datasets[i]

            is_dropped = True
            # This is looking to see if the string is anywhere in the name
            # of the dataset
            for group in groups:
                if group in entry:
                    is_dropped = False
                    break

            if is_dropped:
                if logger is not None:
                    logger.info(f"Not reading out {entry} from the file....")
                datasets.remove(entry)

            i -= 1

        if logger is not None:
            logger.debug(
                f"After only selecting certain datasets ----- " f"datasets: {datasets}"
            )

    return datasets


def validate_subset(
    subset: Union[int, List[int], Tuple[int, int]],
    ncontainers: int,
    logger: Optional[Logger] = None,
) -> List[int]:

    if isinstance(subset, tuple):
        subset_ = list(subset)

    elif isinstance(subset, list):
        subset_ = subset

    elif isinstance(subset, int):
        if logger is not None:
            logger.warning(
                "Single subset value of {subset} being interpreted as a high range"
                f"subset being set to a range of (0,{subset})\n"
            )
        subset_ = [0, subset]
    else:
        raise RuntimeError(f"Unsupported type of subset argument: {subset}")

    # If the user has specified `subset` incorrectly, then let's return
    # an empty data and container
    if subset_[1] - subset_[0] <= 0:
        if logger is not None:
            logger.warning(
                "The range in subset is either 0 or negative!"
                f"{subset_[1]} - {subset_[0]} = {subset_[1] - subset_[0]}"
                "Returning an empty data and container dictionary!\n"
            )
        raise RuntimeError("Range in subset is 0 or negative")

    if subset_[0] > ncontainers:
        if logger is not None:
            logger.error(
                "Range for subset starts greater than number of containers in file!"
                f"{subset_[0]} > {ncontainers}"
            )
        raise RuntimeError(
            "Range for subset starts greater than number of containers in file!"
        )

    if subset_[1] > ncontainers:
        if logger is not None:
            logger.warning(
                "Range for subset is greater than number of containers in file!"
                f"{subset_[1]} > {ncontainers}"
                f"High range of subset will be set to {ncontainers}\n"
            )
        subset_[1] = ncontainers

    if subset_[1] <= subset_[0]:
        if logger is not None:
            logger.error(
                "Will not be reading anything in!"
                f"High range of {subset_[1]} is less than or equal to low range of {subset_[0]}"
            )
        raise RuntimeError(
            f"High range of {subset_[1]} is less than or equal to low range of {subset_[0]}"
        )

    return subset_


def calculate_index_from_counters(counters: Dataset) -> ndarray:
    index = np.add.accumulate(counters) - counters
    return index


def unpack(
    container: Dict[str, Any],
    data: Dict[str, Any],
    n: int = 0,
) -> None:

    """Fills the container dictionary with selected rows from the data dictionary.

    Args:

        **container** (dict): container dictionary to be filled

        **data** (dict): Data dictionary used to fill the container dictionary

    **n** (integer): 0 by default. Which entry should be pulled out of the data
                     dictionary and inserted into the container dictionary.

    """

    keys = container.keys()

    for key in keys:

        # if "num" in key:
        if key in data["_LIST_OF_COUNTERS_"]:
            container[key] = data[key][n]

        elif "INDEX" not in key:
            indexkey = data["_MAP_DATASETS_TO_INDEX_"][key]
            numkey = data["_MAP_DATASETS_TO_COUNTERS_"][key]

            if len(data[indexkey]) > 0:
                index = data[indexkey][n]

            if len(data[numkey]) > 0:
                nobjs = data[numkey][n]
                container[key] = data[key][index : index + nobjs]


def get_ncontainers_in_file(
    filename: str, logger: Optional[Logger] = None
) -> Union[None, int64]:

    """Get the number of containers in the file."""

    with h5py.File(filename, "r+") as f:
        a = f.attrs

        if a.__contains__("_NUMBER_OF_CONTAINERS_"):
            _NUMBER_OF_CONTAINERS_ = a.get("_NUMBER_OF_CONTAINERS_")
            f.close()
            return _NUMBER_OF_CONTAINERS_
        else:
            if logger is not None:
                logger.warning(
                    '\nFile does not contain the attribute, "_NUMBER_OF_CONTAINERS_"\n'
                )
            f.close()
            return None


def get_ncontainers_in_data(data, logger=None) -> Union[None, int64]:

    """Get the number of containers in the data dictionary.

    This is useful in case you've only pulled out subsets of the data

    """

    if not isinstance(data, dict):
        if logger is not None:
            logger.warning(f"{data} is not a dictionary!\n")
        return None

    if "_NUMBER_OF_CONTAINERS_" in list(data.keys()):
        _NUMBER_OF_CONTAINERS_ = data["_NUMBER_OF_CONTAINERS_"]
        return _NUMBER_OF_CONTAINERS_
    else:
        if logger is not None:
            logger.warning(
                '\ndata dictionary does not contain the key, "_NUMBER_OF_CONTAINERS_"\n'
            )
        return None


def get_file_metadata(filename: str) -> Union[None, Dict[str, Any]]:

    """Get the file metadata and return it as a dictionary"""

    f = h5py.File(filename, "r+")

    a = f.attrs

    if len(a) < 1:
        f.close()
        return None

    metadata = {}

    for key in a.keys():
        metadata[key] = a[key]

    f.close()

    return metadata


def print_file_metadata(filename: str):

    """Pretty print the file metadata"""

    metadata = get_file_metadata(filename)

    if metadata is None:
        return None

    output = ""

    keys = list(metadata.keys())

    first_keys_to_print = ["date", "_NUMBER_OF_CONTAINERS_"]

    keys_already_printed = []

    # Print the basics first
    for key in first_keys_to_print:
        if key in keys:
            val = metadata[key]
            output += f"{key:<20s} : {val}\n"
            keys_already_printed.append(key)

    # Print the versions next
    for key in keys:
        if key in keys_already_printed:
            continue

        if key.find("version") >= 0:
            val = metadata[key]
            output += f"{key:<20s} : {val}\n"
            keys_already_printed.append(key)

    # Print the read of the metadata
    for key in keys:
        if key in keys_already_printed:
            continue

        val = metadata[key]
        output += f"{key:<20s} : {val}\n"
        keys_already_printed.append(key)

    return output
