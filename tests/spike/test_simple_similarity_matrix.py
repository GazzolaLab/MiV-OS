import numpy as np
import pytest

from miv.signal.similarity.simple import domain_distance_matrix


def test_domain_distance_matrix():
    # create a sample temporal sequence with shape (3, 4)
    temporal_sequence = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 3, 4, 5]])

    # test time domain
    distance_matrix = domain_distance_matrix(temporal_sequence, "time")
    assert distance_matrix.shape == (4, 4)
    np.testing.assert_allclose(
        distance_matrix,
        [[0, 2, 10, 24], [2, 0, 5, 16], [10, 5, 0, 8], [24, 16, 8, 0]],
        rtol=1e-6,
        atol=1e-6,
    )

    # test frequency domain
    distance_matrix = domain_distance_matrix(temporal_sequence, "frequency")
    assert distance_matrix.shape == (4, 4)
    np.testing.assert_allclose(
        distance_matrix,
        [[0, 6, 30, 72], [6, 0, 15, 48], [30, 15, 0, 24], [72, 48, 24, 0]],
        rtol=1e-6,
        atol=1e-6,
    )

    # test power domain
    distance_matrix = domain_distance_matrix(temporal_sequence, "power")
    assert distance_matrix.shape == (4, 4)
    np.testing.assert_allclose(
        distance_matrix,
        [[0, 1, 8, 21], [1, 0, 7, 20], [8, 7, 0, 13], [21, 20, 13, 0]],
        rtol=1e-6,
        atol=1e-6,
    )

    # test ValueError when sequence is not at least 2D
    with pytest.raises(AssertionError):
        domain_distance_matrix(np.array([1, 2, 3, 4]), "time")
