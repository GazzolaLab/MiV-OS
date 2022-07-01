__all__ = ["SpikeCutout", "ChannelSpikeCutout"]

from typing import List, Optional, Tuple, Union

from dataclasses import dataclass

import numpy as np


@dataclass
class SpikeCutout:
    """SpikeCutout class

    Attributes
    ----------
    cutout : np.ndarray
    sampling_rate : float
    pca_comp_index : int
    """

    def __init__(
        self,
        cutout: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        sampling_rate: float,
        pca_comp_index: int,
    ) -> None:
        self.cutout: np.ndarray = cutout
        self.sampling_rate: float = sampling_rate
        self.pca_comp_index: int = pca_comp_index


class ChannelSpikeCutout:
    CATEGORY_NAMES: tuple[str] = ("uncategorized", "neuronal", "false", "mixed")

    def __init__(
        self,
        cutouts: List[SpikeCutout],
        num_components: int,
        categorized: bool = False,
        category_list: Optional[np.ndarray] = None,
    ):
        self.cutouts: List[SpikeCutout] = cutouts
        self.num_components: int = num_components
        self.categorized: bool = categorized
        self.category_list = category_list if categorized else np.zeros(num_components)
