__all__ = ["SpikeCutout"]

from typing import Optional, Union

import string
from dataclasses import dataclass

import neo
import numpy as np
import quantities as pq


@dataclass
class SpikeCutout:
    """SpikeCutout class

    Attributes
    ----------
    cutout : np.ndarray
    sampling_rate : float
    category : int
        (default = 0)
    """

    cutout: np.ndarray = cutout
    sampling_rate: float = sampling_rate
    category: int = 0
    CATEGORY_NAMES: tuple[str] = ("uncategorized", "neuronal", "false", "mixed")
