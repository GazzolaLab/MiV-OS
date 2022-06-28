__all__ = ["MEAGeometryProtocol"]

import typing
from typing import Any, Iterable, Protocol, Tuple

import matplotlib


class MEAGeometryProtocol(Protocol):
    def get_closest_node(self, x: float, y: float) -> int:
        """Given xy coordinate, return closest node idx"""
        ...

    def get_xy(self, idx: int) -> Tuple[float, float]:
        """Given node index, return xy coordinate"""
        ...

    def save(self, path: str) -> None:
        ...

    def load(self, path: str) -> None:
        ...

    def view(self) -> matplotlib.pyplot.Figure:
        """Simplified view of MEA orientation"""
        ...
