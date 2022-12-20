import numpy as np
import pytest

from miv.mea.grid import GridMEA


def test_init():
    # Test valid input
    nrow = 10
    ncol = 20
    xid = np.arange(nrow * ncol) % ncol
    yid = np.arange(nrow * ncol) // ncol
    mea = GridMEA(nrow, ncol, xid, yid)
    assert mea.nrow == nrow
    assert mea.ncol == ncol
    assert (mea.xid == xid).all()
    assert (mea.yid == yid).all()

    # Test invalid input
    # with pytest.raises(ValueError):
    #    GridMEA(nrow, ncol, xid, yid[:-1])
    # with pytest.raises(ValueError):
    #    GridMEA(nrow, ncol, xid[:-1], yid)
    # with pytest.raises(TypeError):
    #    GridMEA(nrow, ncol, xid.tolist(), yid)
    # with pytest.raises(TypeError):
    #    GridMEA(nrow, ncol, xid, yid.tolist())
