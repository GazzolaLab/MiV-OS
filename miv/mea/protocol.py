__all__ = ["MEAGeometryProtocol"]

import typing
from typing import Any, Iterable, Protocol, Tuple

import matplotlib
import numpy as np

from miv.core.operator import _Chainable
from miv.core.policy import _Runnable


class MEAGeometryProtocol(_Chainable, _Runnable, Protocol):
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

    def map_data(self, vector: np.ndarray, missing_value: float) -> np.ndarray:
        """Map data (1-D array) to MEA (2-D array or N-D array)"""
        ...
