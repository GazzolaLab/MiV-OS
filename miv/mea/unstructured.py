__all__ = ["UnstructuredMEA"]

from typing import Tuple

import matplotlib
import numpy as np

from miv.mea.base import MEAMixin


class UnstructuredMEA(MEAMixin):
    """
    A class representing a grid-based multi-electrode array (MEA).

    Example
    -------
    For example, you could create an instance of the GridMEA class like this:
    literal blocks::

        grid = np.arrange(9).reshape(3, 3)
        mea = GridMEA(grid)
    """

    def __init__(
        self,
        values: np.ndarray,
        coordinates: np.ndarray,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.values = values
        self.coordinates = coordinates

    def map_data(self, vector: np.ndarray, missing_value: float = 0.0) -> np.ndarray:
        """Map data (1-D array) to MEA (2-D array or N-D array)"""
        return self.values, self.coordinates[:, 1], self.coordinates[:, 0]
