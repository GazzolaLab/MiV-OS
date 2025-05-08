from collections.abc import Sequence
import numpy as np


def downsample_average(
    x: Sequence[float],
    y: Sequence[float],
    max_samples: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample x and y by averaging over non-overlapping windows so that
    the returned length is approximately max_samples.

    If len(x) <= max_samples, returns the original data as numpy arrays.

    Args:
        x: Sequence of numeric values (e.g. list or 1D array).
        y: Sequence of numeric values, same length as x.
        max_samples: Desired maximum number of output samples.

    Returns:
        A tuple (x_ds, y_ds) of numpy arrays, each of length ~max_samples.
    """
    arr_x = np.asarray(x)
    arr_y = np.asarray(y)
    size = arr_x.shape[0]
    if size <= max_samples:
        return arr_x, arr_y

    best_n = size // max_samples

    # Number of full windows
    bins = size // best_n

    # Trim off any leftover at the end
    x_trim = arr_x[: bins * best_n].reshape(bins, best_n)
    y_trim = arr_y[: bins * best_n].reshape(bins, best_n)

    # Compute the average in each window
    x_ds = x_trim.mean(axis=1)
    y_ds = y_trim.mean(axis=1)
    return x_ds, y_ds
