from collections import UserList

import numpy as np
import pytest

from miv.core import Spikestamps


def test_spikestamps_init():
    # Test that the data attribute is set correctly
    s = Spikestamps([[1, 2, 3], [4, 5, 6]])
    assert s.data == [[1, 2, 3], [4, 5, 6]]
    assert isinstance(s, UserList)

    # Test that the data attribute is set to an empty list if no iterable is provided
    s = Spikestamps([])
    assert s.data == []


def test_spikestamps_setitem():
    s = Spikestamps([[1, 2, 3], [4, 5, 6]])

    # Test that we can set the value of an existing element
    s[0] = [7, 8, 9]
    assert s[0] == [7, 8, 9]

    # Test that an IndexError is raised if the index is out of range
    with pytest.raises(IndexError):
        s[2] = [10, 11, 12]


def test_spikestamps_insert():
    s = Spikestamps([[1, 2, 3], [4, 5, 6]])

    # Test that we can insert an element at the beginning of the list
    s.insert(0, [7, 8, 9])
    assert s == [[7, 8, 9], [1, 2, 3], [4, 5, 6]]

    # Test that we can insert an element at the end of the list
    s.insert(3, [10, 11, 12])
    assert s == [[7, 8, 9], [1, 2, 3], [4, 5, 6], [10, 11, 12]]

    # Test that an IndexError is raised if the index is out of range
    with pytest.raises(IndexError):
        s.insert(10, [13, 14, 15])


def test_spikestamps_append():
    s = Spikestamps([[1, 2, 3], [4, 5, 6]])

    # Test that we can append an element to the end of the list
    s.append([7, 8, 9])
    assert s == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def test_spikestamps_extend():
    s1 = Spikestamps([[1, 2, 3], [4, 5, 6]])
    s2 = [[7, 8, 9], [10, 11, 12]]

    # Test that we can extend the current object with another Spikestamps object
    s1.extend(s2)
    assert s1 == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

    # Test that we can extend the current object with an iterable of arrays
    s1.extend([[13, 14, 15], [16, 17, 18]])
    assert s1 == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
    ]


@pytest.mark.parametrize(
    "array, tstart, tend, expected_truncated_arr",
    [
        (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 3, 8, np.array([3, 4, 5, 6, 7, 8])),
        (
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            0.3,
            0.8,
            np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        ),
    ],
)
def test_spikestamps_window_truncation(array, tstart, tend, expected_truncated_arr):
    # Test case with integer numbers
    spikestamps = Spikestamps([array])
    truncated_arr = spikestamps.get_view(tstart, tend).data[0]

    np.testing.assert_equal(truncated_arr, expected_truncated_arr)  # Check accuracy
    assert np.all(np.isin(truncated_arr, array))  # Check all elements are included
    np.testing.assert_equal(
        np.sort(truncated_arr), truncated_arr
    )  # Check the result is sorted array
