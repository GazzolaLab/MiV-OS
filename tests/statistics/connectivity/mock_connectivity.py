from dataclasses import dataclass

import numpy as np
import pytest


@dataclass
class MockConnectivity:
    channels = [1, 2, 3]

    def __post_init__(self):
        self.mea_map = np.arange(10).reshape([5, 2])
