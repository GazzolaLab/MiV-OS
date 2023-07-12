__doc__ = """ Base mixin for MEA classes. """

from typing import Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from miv.core.operator import BaseChainingMixin


class MEAMixin(BaseChainingMixin):
    """Base mixin for MEA classes.

    Functional module.

    Comply with the following interface:
        MEAGeometryProtocol
    """

    def __init__(self, tag: str = "mea", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag = tag
        self.runner = None

    def get_xy(self, idx: int) -> Tuple[float, float]:
        """Given node index, return xy coordinate"""
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError

    def view(self) -> plt.Figure:
        """Simplified view of MEA orientation"""
        raise NotImplementedError

    @property
    def output(self):
        return self

    def run(self, **kwargs):
        pass

    def map_data(self, vector: np.ndarray) -> np.ndarray:
        """Map data (1-D array) to MEA (2-D array or N-D array)"""
        raise NotImplementedError

    def set_save_path(self, *args, **kwargs):
        pass
