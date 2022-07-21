__all__ = ["GridMEA"]

from typing import Tuple

import matplotlib
import numpy as np

from miv.mea.protocol import MEAGeometryProtocol


class GridMEA:
    def __init__(self, nrow: int, ncol: int, xid: np.ndarray, yid: np.ndarray):
        self.nrow = nrow
        self.ncol = ncol
        self.xid = xid
        self.yid = yid

    def get_closest_node(self, x: float, y: float) -> int:
        """Given xy coordinate, return closest node idx"""
        raise NotImplementedError

    def get_xy(self, idx: int) -> Tuple[float, float]:
        """Given node index, return xy coordinate"""
        raise NotImplementedError

    def view(self) -> matplotlib.pyplot.Figure:
        """Simplified view of MEA orientation"""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Export MEA information"""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Import MEA from external source"""
        raise NotImplementedError
