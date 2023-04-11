__all__ = ["UnstructuredMEA"]

from typing import Tuple

import matplotlib
import numpy as np
from scipy.interpolate import griddata

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
        ), f"The number of indices ({indices.shape[0]}) and coordinates ({coordinates.shape[0]}) must be the same."
        self.indices = indices
        self.coordinates = coordinates

    def to_json(self) -> dict:
        """Return a JSON-serializable dictionary"""
        return {
            "indices": self.indices.tolist(),
            "coordinates": self.coordinates.tolist(),
        }

    def map_data(
        self, vector: np.ndarray, missing_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map data (1-D array) to MEA (2-D array or N-D array)"""
        values = np.full(self.indices.shape, missing_value)
        for idx, value in enumerate(vector):
            if idx not in self.indices:
                continue
            values[self.indices == idx] = value

        xmin = np.min(self.coordinates[:, 0])
        xmax = np.max(self.coordinates[:, 0])
        ymin = np.min(self.coordinates[:, 1])
        ymax = np.max(self.coordinates[:, 1])
        delx = xmax - xmin
        dely = ymax - ymin
        x = np.linspace(xmin - 0.1 * delx, xmax + 0.1 * delx, 100)
        y = np.linspace(ymin - 0.1 * dely, ymax + 0.1 * dely, 100)
        Xn, Yn = np.meshgrid(x, y)
        Z = griddata(self.coordinates, values, (Xn, Yn), method="cubic")

        return Xn, Yn, Z
