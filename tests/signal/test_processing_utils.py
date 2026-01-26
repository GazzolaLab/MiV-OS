import numpy as np
import pytest

from miv.signal.utils import downsample_average


def test_downsample_average_basic():
    x = np.arange(10)
    y = np.arange(10) * 2
    x_ds, y_ds = downsample_average(x, y, max_samples=5)
    assert len(x_ds) == len(y_ds) == 5
    np.testing.assert_allclose(x_ds, np.array([0.5, 2.5, 4.5, 6.5, 8.5]))
    np.testing.assert_allclose(y_ds, np.array([1.0, 5.0, 9.0, 13.0, 17.0]))


def test_downsample_average_returns_original_when_short():
    x = [1, 2, 3]
    y = [4, 5, 6]
    x_ds, y_ds = downsample_average(x, y, max_samples=5)
    np.testing.assert_array_equal(x_ds, np.array(x))
    np.testing.assert_array_equal(y_ds, np.array(y))


def test_downsample_average_invalid_max_samples():
    with pytest.raises(ValueError, match="max_samples must be positive"):
        downsample_average([1, 2], [1, 2], max_samples=0)


def test_downsample_average_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        downsample_average([1, 2, 3], [1, 2], max_samples=2)


def test_downsample_average_empty_inputs():
    """Test that downsample_average handles empty inputs correctly."""
    x = np.array([])
    y = np.array([])
    x_ds, y_ds = downsample_average(x, y, max_samples=5)
    assert len(x_ds) == 0
    assert len(y_ds) == 0
    assert isinstance(x_ds, np.ndarray)
    assert isinstance(y_ds, np.ndarray)


def test_downsample_average_single_element():
    """Test that downsample_average handles single-element inputs correctly."""
    x = np.array([5.0])
    y = np.array([10.0])
    x_ds, y_ds = downsample_average(x, y, max_samples=5)
    assert len(x_ds) == 1
    assert len(y_ds) == 1
    np.testing.assert_array_equal(x_ds, np.array([5.0]))
    np.testing.assert_array_equal(y_ds, np.array([10.0]))
