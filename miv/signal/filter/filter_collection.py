__doc__ = """
Filter collection.
"""
__all__ = ["FilterCollection"]

from typing import Union
import numpy as np
import numpy.typing as npt

from collections.abc import Callable, MutableSequence


class FilterCollection(MutableSequence, Callable):
    def __call__(self, array: npt.ArrayLike) -> np.ndarray:
        raise NotImplementedError  # TODO
