__all__ = ["GridMEA"]

from typing import Tuple

import matplotlib
import numpy as np

from miv.mea.base import MEAMixin


class GridMEA(MEAMixin):
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
        assert grid.ndim == 2, "The grid must be 2-D."
        assert spacing[0] > 0 and spacing[1] > 0, "The spacing must be positive."
        self.grid = grid
        self.spacing = spacing

        self.nrow, self.ncol = grid.shape

    def map_data(
        self, vector: np.ndarray, missing_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map data (1-D array) to MEA (2-D array or N-D array)"""
        value_grid = np.full_like(self.grid, missing_value)
        for idx, value in enumerate(vector):
            if idx not in self.grid:
                continue
            value_grid[self.grid == idx] = value
        X = np.arange(self.ncol) * self.spacing[0]
        Y = np.arange(self.nrow) * self.spacing[1]
        Xn, Yn = np.meshgrid(X, Y)
        return Xn, Yn, value_grid
