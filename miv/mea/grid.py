__all__ = ["GridMEA"]

from typing import Tuple

import matplotlib

from miv.mea.protocol import MEAGeometryProtocol


class GridMEA:
    def get_closest_node(self, x: float, y: float) -> int:
        """Given xy coordinate, return closest node idx"""
        raise NotImplementedError

    def get_xy(self, idx: int) -> Tuple[float, float]:
        """Given node index, return xy coordinate"""
        raise NotImplementedError

    def view(self) -> matplotlib.pyplot.Figure:
        """Simplified view of MEA orientation"""
        raise NotImplementedError
