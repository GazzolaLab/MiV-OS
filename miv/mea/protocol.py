__all__ = ["MEAGeometryProtocol"]

import typing
from typing import Any, Iterable, Protocol, Tuple

import matplotlib
import numpy as np

from miv.core.operator.cachable import _Jsonable
from miv.core.operator.chainable import _Chainable
from miv.core.policy import _Runnable


class MEAGeometryProtocol(_Jsonable, _Chainable, _Runnable, Protocol):
    @property
    def coordinates(self) -> np.ndarray:
        """Return coordinates of MEA electrodes location"""
        ...

    def get_xy(self, idx: int) -> Tuple[float, float]:
        """Given node index, return xy coordinate"""
        ...

    def get_ixiy(self, idx: int) -> Tuple[int, int]:
        """Given node index, return coordinate index"""
        ...

    def save(self, path: str) -> None:
        ...

    def load(self, path: str) -> None:
        ...

    def view(self):
        """Simplified view of MEA orientation"""
        ...

    def map_data(self, vector: np.ndarray, missing_value: float) -> np.ndarray:
        """Map data (1-D array) to MEA (2-D array or N-D array)"""
        ...
