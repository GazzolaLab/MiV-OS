__all__ = ["GridMEA"]

from typing import Tuple

import matplotlib
import numpy as np

from miv.core.datatype import Signal
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
        spacing: Tuple[float, float] = (200, 200),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert grid.ndim == 2, "The grid must be 2-D."
        assert spacing[0] > 0 and spacing[1] > 0, "The spacing must be positive."
        self.grid = grid
        self.spacing = spacing

        self.nrow, self.ncol = grid.shape
        X = np.arange(self.ncol) * self.spacing[0]
        Y = np.arange(self.nrow) * self.spacing[1]
        self.Xn, self.Yn = np.meshgrid(X, Y)

    def to_json(self) -> dict:
        """Return a JSON-serializable dictionary"""
        return {
            "grid": self.grid.tolist(),
            "spacing": self.spacing,
        }

    def map_data(
        self, vector: np.ndarray, missing_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map data (1-D array) to MEA (2-D array or N-D array)"""
        value_grid = np.full_like(self.grid, missing_value, dtype=np.float_)
        for idx, value in enumerate(vector):
            if idx not in self.grid:
                continue
            value_grid[self.grid == idx] = value

        return self.Xn, self.Yn, value_grid

    def map_temporal_data(self, signal: Signal, missing_value: float = 0.0):
        """Map signal data to MEA"""
        n_time = signal.shape[signal._SIGNALAXIS]
        value_grid = np.full(
            [n_time, self.nrow, self.ncol], missing_value, dtype=np.float_
        )
        for idx in range(signal.number_of_channels):
            if idx not in self.grid:
                continue
            value_grid[:, self.grid == idx] = signal[idx][:, None]
        return self.Xn, self.Yn, value_grid

    def get_ixiy(self, channel: int):
        """Given node index, return x y coordinate index"""
        if channel not in self.grid:
            return None
        ys, xs = np.where(self.grid == channel)
        assert len(xs) == 1 and len(ys) == 1, f"The index {channel} is not unique."
        return ys[0], xs[0]

    @property
    def coordinates(self):
        """Return the coordinates of the electrodes

        https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        """

        return np.vstack([self.Xn.ravel(), self.Yn.ravel()]).T
