__doc__ = """ Base mixin for MEA classes. """


import matplotlib.pyplot as plt
import numpy as np

from miv.core.operator.operator import BaseChainingMixin
from miv.core.operator.loggable import DefaultLoggerMixin


class MEAMixin(BaseChainingMixin, DefaultLoggerMixin):
    """Base mixin for MEA classes.

    Functional module.

    Comply with the following interface:
        MEAGeometryProtocol
    """

    def __init__(self, tag: str = "mea", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag = tag
        self.runner = None

    def get_xy(self, idx: int) -> tuple[float, float]:
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
        return self.output

    def map_data(self, vector: np.ndarray) -> np.ndarray:
        """Map data (1-D array) to MEA (2-D array or N-D array)"""
        raise NotImplementedError

    def set_save_path(self, *args, **kwargs):
        pass
