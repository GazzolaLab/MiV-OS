__doc__ = """
Behavior definition of all filter operator implemented in this directory.
"""
__all__ = ["FilterProtocol"]

from typing import Protocol

import numpy as np
import numpy.typing as npt


class FilterProtocol(Protocol):
    def __call__(self, array: npt.ArrayLike) -> np.ndarray:
        ...
