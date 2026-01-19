from collections.abc import Sequence

import numpy as np
import pytest

from miv.core import Spikestamps


def test_spikestamps_init():
    # Test that the data attribute is set correctly
    s = Spikestamps(np.array([[-5, -0.4, 0.2, 2, 3], [-3, 0.1, 4, 5, 6]]) + 0.5)
    ti, tf = 0, 10
    duration = 1.0
    step_size = 0.2
    views = list(s.get_view_stream(ti, tf, step_size=step_size, duration=duration))

    assert len(views) == int(np.ceil((tf - ti) / step_size))
    np.testing.assert_allclose(views[0].data[0], [0.1, 0.7])
    np.testing.assert_allclose(views[0].data[1], [0.6])
    np.testing.assert_allclose(views[1].data[0], [0.7])
    np.testing.assert_allclose(views[1].data[1], [0.6])
