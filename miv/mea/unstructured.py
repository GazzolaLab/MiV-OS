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
        grid: np.ndarray,
        spacing: Tuple[float, float] = (100, 100),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.grid = grid
        self.spacing = spacing

        self.nrow, self.ncol = grid.shape

    def map_data(self, vector: np.ndarray, missing_value: float = 0.0) -> np.ndarray:
        """Map data (1-D array) to MEA (2-D array or N-D array)"""
        value_grid = np.empty_like(self.grid, dtype=vector.dtype)
        value_grid[:] = missing_value
        for idx, value in enumerate(vector):
            r, c = np.where([self.grid == idx])[0]
            value_grid[r, c] = value
        return value_grid
