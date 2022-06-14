from scipy.io import loadmat

from miv.datasets.criticality import load_data


def test_detect_avalanche():
    data = loadmat(load_data())
    assert len(data.keys()) > 0
