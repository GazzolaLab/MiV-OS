__all__ = ["GridMEA"]

from typing import Tuple

import matplotlib
import MEAutility
import numpy as np

from miv.core.datatype import Signal
from miv.mea.base import MEAMixin


class GridMEA(MEAMixin, MEAutility.core.MEA):
    """
    A class representing a grid-based multi-electrode array (MEA).

    Example
    -------
    The best way to create MEA module is through `miv.mea.MEA` object.
    For example, you could create an instance of the GridMEA class like this:
    literal blocks::

        mea: GridMEA = miv.mea.MEA.return_mea("64_intanRHD")
    """

    def __init__(
        self,
        positions,  # Exclude all "None" electrodes
        *args,
        **kwargs,
    ):
        positions = positions.astype(np.float_)
        info = kwargs["info"]

        # Remove nan channels - in json, nan channels are open or broken channels
        MEAMixin.__init__(
            self, positions=positions[~np.isnan(positions).any(axis=1)], *args, **kwargs
        )

        self.grid = np.zeros(info["dim"], dtype=int) - 1
        for channel in range(positions.shape[0]):
            x, y, _ = positions[channel]
            if np.isnan(x) or np.isnan(y):
                continue
            ix = int(np.round(x / self.pitch[0]))
            iy = int(np.round(y / self.pitch[1]))
            self.grid[iy, ix] = channel
        self.grid = self.grid[::-1, :]

        self.nrow, self.ncol = self.grid.shape
        self.X = np.linspace(0, self.pitch[0] * self.ncol, self.ncol)
        self.Y = np.linspace(0, -self.pitch[1] * self.nrow, self.nrow)
        self.Xn, self.Yn = np.meshgrid(self.X, self.Y)

    @property
    def channels(self):
        return sorted(self.grid[self.grid >= 0].ravel().tolist())

    def to_json(self) -> dict:
        """Return a JSON-serializable dictionary"""
        return {
            "grid": self.grid.tolist(),
            "pitch": self.pitch,
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

    def figsize(self, scale=(4, 4)):
        """Return an ideal figure size for plt.figure"""
        return (self.ncol * scale[0], self.nrow * scale[1])

    # Override: get_electrode_matrix
    def get_electrode_matrix(self):
        return self.grid
