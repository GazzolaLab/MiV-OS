__all__ = ["SpikeCutout", "ChannelSpikeCutout"]

from typing import Any, Dict, List, Optional, Tuple, Union

from dataclasses import dataclass

import numpy as np


@dataclass
class SpikeCutout:
    """SpikeCutout class

    Attributes
    ----------
    cutout : Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
    sampling_rate : float
    extractor_comp_index : int
    time : float
    """

    def __init__(
        self,
        cutout: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        sampling_rate: float,
        extractor_comp_index: int,
        time: float,
    ) -> None:
        self.cutout: np.ndarray = np.array(cutout)
        self.sampling_rate: float = sampling_rate
        self.extractor_comp_index: int = extractor_comp_index
        self.time: float = time
        self.categorized: bool = False

    def __len__(self) -> int:
        return len(self.cutout)


class ChannelSpikeCutout:
    """This class holds the SpikeCutout objects for a single channel

    Attributes
    ----------
    cutouts : np.ndarray
        1D NumPy array of SpikeCutout objects that belong to the same channel
    extractor_decomposition_prarmeter : int
        Number of components for PCA decomposition or the number of features
        for wavelet decomposition
    channel_index : int
    categorized : bool
    categorization_list : np.ndarray, defualt = np.array([])
        List of categorization
        (categorization_list[component index][category index])
    """

    CATEGORY_NAMES = ["neuronal", "false", "uncategorized"]

    def __init__(
        self,
        cutouts: np.ndarray,
        num_components: int,
        channel_index: int,
        categorization_list: np.ndarray = np.array([]),
    ):
        self.cutouts: np.ndarray = cutouts
        self.num_components: int = num_components
        self.channel_index: int = channel_index
        valid_cat_list = (len(categorization_list) == num_components) and (
            -1 not in categorization_list
        )
        self.categorization_list: np.ndarray = (
            np.array(categorization_list)
            if valid_cat_list
            else -1 * np.ones(num_components)
        )
        self.categorize(self.categorization_list)

    def __len__(self) -> int:
        return len(self.cutouts)

    def get_cutouts_by_component(self) -> List[List[SpikeCutout]]:
        """
        Returns
        -------
        2D list of SpikeCutout elements where rows correspond to extractor component indices
        """
        components: List[List[SpikeCutout]] = []
        for row_index in range(self.num_components):
            components.append([])
        for index, cutout in enumerate(self.cutouts):
            components[cutout.extractor_comp_index].append(cutout)
        return components

    def categorize(self, category_indices: np.ndarray) -> None:
        """
        Categorize the components in this channel with category indices in
        a 1D list where each element corresponds to the component index.
        Note: if any spike is not categorized, then this channel will not be labeled as categorized.
        Note: if category_indices parameter length does not match the number of groups of extracted spikes,
        then all cutouts are labeled as "uncategorized."

        CATEGORY_NAMES = ["neuronal", "false", "uncategorized"]

        Parameters
        ----------
        category_indices : np.ndarray
            The indices of the category for each group of extracted spikes

            Example:
            categorize(np.array([0, 1, -1])) categorizes component 0 as neuronal spikes,
            component 1 as false spikes, and component 2 as uncategorized.
        """

        if len(category_indices) != self.num_components:
            self.categorization_list = -1 * np.ones(self.num_components, dtype=int)
            self.categorized = False
            return

        self.categorization_list = category_indices
        self.categorized = -1 not in self.categorization_list

        for cutout_index, cutout in enumerate(self.cutouts):
            cutout.categorized = (
                self.categorization_list[cutout.extractor_comp_index] != -1
            )

    def get_labeled_cutouts(self) -> Dict[str, Any]:
        """
        This function returns only the cutouts that were categorized.

        Returns
        -------
        labels :
            1D Numpy array of category label index for each spike
        cutouts :
            1D list of corresponding cutouts
        size :
            int value for the number of labeled cutouts
        """
        labels = []
        labeled_cutouts = []
        size = 0
        for cutout_index, cutout in enumerate(self.cutouts):
            if cutout.categorized:
                labels.append(self.categorization_list[cutout.extractor_comp_index])
                labeled_cutouts.append(list(cutout.cutout))
                size += 1
        return {
            "labels": np.array(labels),
            "cutouts": np.array(labeled_cutouts),
            "size": size,
        }
