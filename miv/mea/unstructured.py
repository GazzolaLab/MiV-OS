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
        indices: np.ndarray,
        coordinates: np.ndarray,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert (
            indices.shape[0] == coordinates.shape[0]
        ), "The number of indices and coordinates must be the same."
        self.indices = indices
        self.coordinates = coordinates

    def map_data(
        self, vector: np.ndarray, missing_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map data (1-D array) to MEA (2-D array or N-D array)"""
        values = np.full(self.indices.shape, missing_value)
        for idx, value in enumerate(vector):
            if idx not in self.indices:
                continue
            values[self.indices == idx] = value

        return values, self.coordinates[:, 1], self.coordinates[:, 0]
