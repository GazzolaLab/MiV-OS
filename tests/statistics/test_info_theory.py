import numpy as np
import pytest

from miv.core.datatype import Spikestamps
from miv.statistics import (
    active_information,
    block_entropy,
    conditional_entropy,
    entropy_rate,
    mutual_information,
    probability_distribution,
    relative_entropy,
    shannon_entropy,
    transfer_entropy,
)


@pytest.fixture
def spikestamps():
    return Spikestamps(
        [
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) + 1,
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) + 2,
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) + 3,
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) + 4,
        ]
    )


def test_probability_distribution(spikestamps):
    output = probability_distribution(spikestamps, 1, 0, 10)
    expected = np.array(
        [
            [1.0, 0.1, 0.2, 0.3, 0.4],
            [1.0, 0.9, 0.2, 0.3, 0.4],
            [1.0, 0.9, 0.8, 0.3, 0.4],
            [1.0, 0.9, 0.8, 0.7, 0.4],
            [1.0, 0.9, 0.8, 0.7, 0.6],
            [1.0, 0.9, 0.8, 0.7, 0.6],
            [1.0, 0.9, 0.8, 0.7, 0.6],
            [1.0, 0.9, 0.8, 0.7, 0.6],
            [1.0, 0.9, 0.8, 0.7, 0.6],
            [1.0, 0.9, 0.8, 0.7, 0.6],
        ]
    )
    np.testing.assert_allclose(output, expected)


def test_shannon_entropy_output(spikestamps):
    output = shannon_entropy(spikestamps, 1, 0, 10)
    expected = np.array(
        [
            [
                np.nan,
                0.468995593589281,
                0.721928094887362,
                0.881290899230693,
                0.970950594454669,
            ]
        ]
    )
    np.testing.assert_allclose(output, expected)


# def test_block_entropy_output():
#    with np.testing.assert_raises(AssertionError):
#        output = block_entropy(spikestamps, 0, 1, 0, 0, 0.1)
#    with np.testing.assert_raises(AssertionError):
#        output = block_entropy(spikestamps, 0, 1, 0, 1, 0)
#    output = block_entropy(spikestamps, 0, 4, 0, 1, 1)
#    np.testing.assert_allclose(output, 0.0)
#
#
# def test_entropy_rate_output():
#    with np.testing.assert_raises(AssertionError):
#        output = entropy_rate(spikestamps, 0, 1, 0, 0, 0.1)
#    with np.testing.assert_raises(AssertionError):
#        output = entropy_rate(spikestamps, 0, 1, 0, 1, 0)
#    output = entropy_rate(spikestamps, 0, 1, 0, 1, 1)
#    np.testing.assert_allclose(output, 0.0)
#
#
# def test_active_information_output():
#    with np.testing.assert_raises(AssertionError):
#        output = active_information(spikestamps, 0, 1, 0, 0, 0.1)
#    with np.testing.assert_raises(AssertionError):
#        output = active_information(spikestamps, 0, 1, 0, 1, 0)
#    output = active_information(spikestamps, 0, 1, 0, 1, 1)
#    np.testing.assert_allclose(output, 0.0)
#
#
# def test_mutual_information_output():
#    with np.testing.assert_raises(AssertionError):
#        output = mutual_information(spikestamps, 0, 0, 0, 0, 0.1)
#    with np.testing.assert_raises(AssertionError):
#        output = mutual_information(spikestamps, 0, 0, 0, 1, 0)
#    output = mutual_information(spikestamps, 0, 0, 0, 1, 1)
#    np.testing.assert_allclose(output, 0.0)
#
#
# def test_relative_entropy_output():
#    with np.testing.assert_raises(AssertionError):
#        output = relative_entropy(spikestamps, 0, 0, 0, 0, 0.1)
#    with np.testing.assert_raises(AssertionError):
#        output = relative_entropy(spikestamps, 0, 0, 0, 1, 0)
#    output = relative_entropy(spikestamps, 0, 0, 0, 1, 1)
#    np.testing.assert_allclose(output, 0.0)
#
#
# def test_conditional_entropy_output():
#    with np.testing.assert_raises(AssertionError):
#        output = conditional_entropy(spikestamps, 0, 0, 0, 0, 0.1)
#    with np.testing.assert_raises(AssertionError):
#        output = conditional_entropy(spikestamps, 0, 0, 0, 1, 0)
#    output = conditional_entropy(spikestamps, 0, 0, 0, 1, 1)
#    np.testing.assert_allclose(output, 0.0)
#
#
# def test_transfer_entropy_output():
#    with np.testing.assert_raises(AssertionError):
#        output = transfer_entropy(spikestamps, 0, 0, 1, 0, 0, 0.1)
#    with np.testing.assert_raises(AssertionError):
#        output = transfer_entropy(spikestamps, 0, 0, 1, 0, 1, 0)
#    output = transfer_entropy(spikestamps, 0, 0, 1, 0, 1, 1)
#    np.testing.assert_allclose(output, 0.0)
