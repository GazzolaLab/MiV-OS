__all__ = ["SpikeCutout", "ChannelSpikeCutout"]

from typing import List, Optional, Tuple, Union

from dataclasses import dataclass

import numpy as np


@dataclass
class SpikeCutout:
    """SpikeCutout class

    Attributes
    ----------
    cutout : Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
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
    """This class holds the SpikeCutout objects for a single channel

    Attributes
    ----------
    cutouts : List[Spikecutout]
        List of SpikeCutout objects that belong to the same channel
    num_components : int
        Number of components for PCA decomposition
    channel_index : int
    categorized : bool
    categorization_list : Optional[np.ndarray], defualt = None
        List of categorization
        (categorization_list[component index][category index])
    """

    CATEGORY_NAMES = ["uncategorized", "neuronal", "false", "mixed"]

    def __init__(
        self,
        cutouts: List[SpikeCutout],
        num_components: int,
        channel_index: int,
        categorization_list: Optional[np.ndarray] = None,
    ):
        self.cutouts: List[SpikeCutout] = cutouts
        self.num_components: int = num_components
        self.channel_index: int = channel_index
        self.categorized: bool = categorization_list
        self.categorization_list = (
            categorization_list if self.categorized else np.zeros(num_components)
        )

    def categorize(self, category_index: List[int]) -> None:
        self.categorization_list = category_index
        if 0 not in self.categorization_list:
            self.categorized = True
