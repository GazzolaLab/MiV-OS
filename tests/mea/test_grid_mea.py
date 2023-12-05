import numpy as np
import pytest

from miv.mea.grid import GridMEA


def test_init():
    # Test valid input
    nrow = 1
    ncol = 2
    info = {
        "dim": [nrow, ncol],
        "pitch": [1, 1],
        "pos": [[0, 0], [1, 0]],
    }

    positions = np.array([[0, 0, 0], [1, 0, 0]])

    mea = GridMEA(positions=positions, info=info)
    assert mea.nrow == nrow
    assert mea.ncol == ncol
