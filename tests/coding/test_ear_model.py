import numpy as np
import pytest

from miv.coding.temporal import LyonEarModel


def test_ear_model_simple_array():
    # Case from https://engineering.purdue.edu/~malcolm/interval/1998-010/
    ear_model = LyonEarModel(400, 1)
    inputs = np.array([1, 0, 0, 0, 0, 0], dtype=np.double)
    out = ear_model(inputs)

    np.testing.assert_allclose(
        out,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.16309133, 0.09075808],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.01684571, 0.0],
            ]
        ),
    )
