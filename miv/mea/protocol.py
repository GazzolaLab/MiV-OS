__all__ = ["MEAGeometryProtocol"]

from typing import Protocol

import numpy as np

from miv.core.protocol import _Jsonable
from miv.core.operator.protocol import _Chainable


class MEAGeometryProtocol(_Jsonable, _Chainable, Protocol):
    @property
    def coordinates(self) -> np.ndarray:
        """Return coordinates of MEA electrodes location"""
        ...

    def get_xy(self, idx: int) -> tuple[float, float]:
        """Given node index, return xy coordinate"""
        ...

    def get_ixiy(self, idx: int) -> tuple[int, int]:
        """Given node index, return coordinate index"""
        ...

    def save(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...

    def view(self):
        """Simplified view of MEA orientation"""
        ...

    def map_data(self, vector: np.ndarray, missing_value: float) -> np.ndarray:
        """Map data (1-D array) to MEA (2-D array or N-D array)"""
        ...
