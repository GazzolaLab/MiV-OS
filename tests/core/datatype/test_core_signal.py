from typing import List, Tuple

import numpy as np
import pytest

from miv.core.datatype import Signal


@pytest.fixture
def signal_data() -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Returns an example signal data to be used in tests.
    """
    data = np.array([[1, 2, 3], [4, 5, 6]])
    timestamps = np.array([0, 1, 2])
    rate = 1000
    return data, timestamps, rate


@pytest.fixture
def signal_object(signal_data) -> Signal:
    """
    Returns an example signal object to be used in tests.
    """
    data, timestamps, rate = signal_data
    return Signal(data=data, timestamps=timestamps, rate=rate)


def test_number_of_channels(signal_object):
    assert signal_object.number_of_channels == 3


def test_getitem(signal_object):
    assert np.array_equal(signal_object[1], np.array([2, 5]))


def test_select(signal_object):
    new_signal = signal_object.select((0, 2))
    assert np.array_equal(new_signal.data, np.array([[1, 3], [4, 6]]))


def test_get_start_time(signal_object):
    assert signal_object.get_start_time() == 0


def test_get_end_time(signal_object):
    assert signal_object.get_end_time() == 2


def test_shape(signal_object):
    assert signal_object.shape == (2, 3)


def test_append(signal_object):
    new_channel = np.array([[7], [8]])
    signal_object.append(new_channel)
    assert np.array_equal(signal_object.data, np.array([[1, 2, 3, 7], [4, 5, 6, 8]]))


def test_extend_signal(signal_object):
    new_data = np.array([[7, 8, 9], [10, 11, 12]])
    new_time = np.array([3, 4])
    signal_object.extend_signal(new_data, new_time)

    result_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    result_time = np.array([0, 1, 2, 3, 4])
    assert np.array_equal(signal_object.data, result_data)
    assert np.array_equal(signal_object.timestamps, result_time)


def test_prepend_signal(signal_object):
    new_data = np.array([[7, 8, 9], [10, 11, 12]])
    new_time = np.array([3, 4])
    signal_object.prepend_signal(new_data, new_time)
    assert np.array_equal(
        signal_object.data, np.array([[7, 8, 9], [10, 11, 12], [1, 2, 3], [4, 5, 6]])
    )
    assert np.array_equal(signal_object.timestamps, np.array([0, 1, 2, 3, 4]))


# Negative Tests
def test_signal_negative():
    # Test invalid signal initialization
    with pytest.raises(AssertionError):
        Signal([1, 2, 3], [0, 1, 2])

    # Test invalid indexing
    s = Signal(np.ones((10, 3)), np.arange(10))
    with pytest.raises(IndexError):
        s[10]

    # Test invalid channel selection
    s = Signal(np.ones((10, 3)), np.arange(10))
    with pytest.raises(IndexError):
        s.select((0, 1, 2, 3))

    # Test appending invalid signal
    s = Signal(np.ones((10, 3)), np.arange(10))
    with pytest.raises(AssertionError):
        s.append(np.ones((5, 4)))

    # Test extending signal with invalid input
    s = Signal(np.ones((10, 3)), np.arange(10))
    with pytest.raises(AssertionError):
        s.extend_signal(np.ones((10, 2)), np.arange(10))

    # Test prepending signal with invalid input
    s = Signal(np.ones((10, 3)), np.arange(10))
    with pytest.raises(AssertionError):
        s.prepend_signal(np.ones((10, 2)), np.arange(10))
