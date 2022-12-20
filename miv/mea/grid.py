__all__ = ["GridMEA"]

from typing import Tuple

import matplotlib
import numpy as np

from miv.mea.protocol import MEAGeometryProtocol


class GridMEA:
    """
    A class representing a grid-based multi-electrode array (MEA).

    Example
    -------
    For example, you could create an instance of the GridMEA class like this:
    literal blocks::

        nrow = 10
        ncol = 20
        xid = np.arange(nrow * ncol) % ncol
        yid = np.arange(nrow * ncol) // ncol
        mea = GridMEA(nrow, ncol, xid, yid)

    Attributes
    ----------
    nrow: int
        The number of rows in the grid.
    ncol: int
        The number of columns in the grid.
    xid: np.ndarray
        A NumPy array containing the x-coordinates of the electrodes in the grid.
    yid: np.ndarray
        A NumPy array containing the y-coordinates of the electrodes in the grid.
    """

    def __init__(self, nrow: int, ncol: int, xid: np.ndarray, yid: np.ndarray):
        self.nrow = nrow
        self.ncol = ncol
        self.xid = xid
        self.yid = yid

    def get_closest_node(self, x: float, y: float) -> int:  # pragma: no cover
        """Given xy coordinate, return closest node idx"""
        raise NotImplementedError

    def get_xy(self, idx: int) -> Tuple[float, float]:  # pragma: no cover
        """Given node index, return xy coordinate"""
        raise NotImplementedError

    def view(self) -> matplotlib.pyplot.Figure:  # pragma: no cover
        """Simplified view of MEA orientation"""
        raise NotImplementedError

    def save(self, path: str) -> None:  # pragma: no cover
        """Export MEA information"""
        raise NotImplementedError

    def load(self, path: str) -> None:  # pragma: no cover
        """Import MEA from external source"""
        raise NotImplementedError
