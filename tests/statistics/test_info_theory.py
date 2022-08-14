import numpy as np
import pytest
from neo.core import Segment, SpikeTrain

# Test set For Info_Theory module
from miv.statistics import (
    active_information,
    block_entropy,
    conditional_entropy,
    entropy_rate,
    mutual_information,
    relative_entropy,
    shannon_entropy,
    transfer_entropy,
)

seg = Segment(index=1)
train0 = SpikeTrain(
    times=[0, 1, 2],
    units="sec",
    t_stop=3,
)
seg.spiketrains.append(train0)


def test_shannon_entropy_output():
    with np.testing.assert_raises(AssertionError):
        output = shannon_entropy(seg.spiketrains, 0, 0, 0, 0.1)
    with np.testing.assert_raises(AssertionError):
        output = shannon_entropy(seg.spiketrains, 0, 0, 1, 0)
    output = shannon_entropy(seg.spiketrains, 0, 0, 1, 1)
    np.testing.assert_allclose(output, 1.0)


def test_block_entropy_output():
    with np.testing.assert_raises(AssertionError):
        output = block_entropy(seg.spiketrains, 0, 1, 0, 0, 0.1)
    with np.testing.assert_raises(AssertionError):
        output = block_entropy(seg.spiketrains, 0, 1, 0, 1, 0)
    output = block_entropy(seg.spiketrains, 0, 1, 0, 1, 1)
    np.testing.assert_allclose(output, 0.0)


def test_entropy_rate_output():
    with np.testing.assert_raises(AssertionError):
        output = entropy_rate(seg.spiketrains, 0, 1, 0, 0, 0.1)
    with np.testing.assert_raises(AssertionError):
        output = entropy_rate(seg.spiketrains, 0, 1, 0, 1, 0)
    output = entropy_rate(seg.spiketrains, 0, 1, 0, 1, 1)
    np.testing.assert_allclose(output, 0.0)


def test_active_information_output():
    with np.testing.assert_raises(AssertionError):
        output = active_information(seg.spiketrains, 0, 1, 0, 0, 0.1)
    with np.testing.assert_raises(AssertionError):
        output = active_information(seg.spiketrains, 0, 1, 0, 1, 0)
    output = active_information(seg.spiketrains, 0, 1, 0, 1, 1)
    np.testing.assert_allclose(output, 0.0)


def test_mutual_information_output():
    with np.testing.assert_raises(AssertionError):
        output = mutual_information(seg.spiketrains, 0, 0, 0, 0, 0.1)
    with np.testing.assert_raises(AssertionError):
        output = mutual_information(seg.spiketrains, 0, 0, 0, 1, 0)
    output = mutual_information(seg.spiketrains, 0, 0, 0, 1, 1)
    np.testing.assert_allclose(output, 0.0)


def test_relative_entropy_output():
    with np.testing.assert_raises(AssertionError):
        output = relative_entropy(seg.spiketrains, 0, 0, 0, 0, 0.1)
    with np.testing.assert_raises(AssertionError):
        output = relative_entropy(seg.spiketrains, 0, 0, 0, 1, 0)
    output = relative_entropy(seg.spiketrains, 0, 0, 0, 1, 1)
    np.testing.assert_allclose(output, 0.0)


def test_conditional_entropy_output():
    with np.testing.assert_raises(AssertionError):
        output = conditional_entropy(seg.spiketrains, 0, 0, 0, 0, 0.1)
    with np.testing.assert_raises(AssertionError):
        output = conditional_entropy(seg.spiketrains, 0, 0, 0, 1, 0)
    output = conditional_entropy(seg.spiketrains, 0, 0, 0, 1, 1)
    np.testing.assert_allclose(output, 0.0)


def test_transfer_entropy_output():
    with np.testing.assert_raises(AssertionError):
        output = transfer_entropy(seg.spiketrains, 0, 0, 1, 0, 0, 0.1)
    with np.testing.assert_raises(AssertionError):
        output = transfer_entropy(seg.spiketrains, 0, 0, 1, 0, 1, 0)
    output = transfer_entropy(seg.spiketrains, 0, 0, 1, 0, 1, 1)
    np.testing.assert_allclose(output, 0.0)
