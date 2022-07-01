__all__ = ["SpikeCutout"]

from dataclasses import dataclass

import numpy as np


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

    CATEGORY_NAMES: tuple[str] = ("uncategorized", "neuronal", "false", "mixed")

    def __init__(
        self, cutout: np.ndarray, sampling_rate: float, category: int = 0
    ) -> None:
        self.cutout: np.ndarray = cutout
        self.sampling_rate: float = sampling_rate
        self.category: int = category
