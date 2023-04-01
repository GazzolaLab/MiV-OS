import numpy as np
import pytest

from miv.mea.grid import GridMEA


def test_init():
    # Test valid input
    nrow = 10
    ncol = 20
    grid = np.arange(nrow * ncol).reshape(nrow, ncol)
    mea = GridMEA(grid)
    assert mea.nrow == nrow
    assert mea.ncol == ncol
