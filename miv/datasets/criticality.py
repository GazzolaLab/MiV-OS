__doc__ = """Sample data for criticality analysis."""
__all__ = ["load_data"]

import gzip
import os

import numpy as np

from miv.datasets.utils import get_file


def load_data():  # pragma: no cover
    """
    Loads the sample for criticality analysis

    Total size: 5.1 MB (compressed)

    Returns
    -------
    datapath: str

    Examples
    --------
        >>> from miv import datasets
        >>> data = datasets.criticality.load_data()
    """

    subdir = "criticality"
    base_url = "https://uofi.box.com/shared/static/mxj6mfjj0fdexod4nqoqovdeg7t8dbbq.mat"
    file = "asdf.mat"
    file_hash = "c8ff82f4700eed2023d37cb7643d24b37350e521469148406042ef206c3aac34"

    path = get_file(
        file_url=base_url,
        directory=subdir,
        fname=file,
        file_hash=file_hash,
        archive_format=None,
    )
    return path
