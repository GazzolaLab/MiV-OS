import numpy as np
import pytest

from miv.visualization.connectivity import (
    plot_connectivity,
    plot_connectivity_interactive,
)


def test_plot_connectivity_output():

    # Initialize the spiketrain as below
    mea_map = np.array([[1, 2], [3, 4], [5, 6]])
    mea_map1 = np.array([[0, 0], [0, 0], [0, 0]])
    connectivity_matrix1 = np.array([[2, 3], [10, 11]])
    connectivity_matrix2 = np.array([[2, 3], [5, 8], [9, 10]])

    with pytest.raises(AssertionError):
        plot_connectivity(mea_map, connectivity_matrix1, False)
    # The function above should throw an error since connectivity matrix is not of same size as mea_map
    with pytest.raises(AssertionError):
        plot_connectivity(mea_map, connectivity_matrix2, False)
    # The function above should throw an error since connectivity matrix is not a square matrix
    with pytest.raises(AssertionError):
        plot_connectivity(mea_map1, connectivity_matrix2, True)
        # The function above should throw an error since MEA map contains no identification


def test_plot_connectivity_interactive_output():

    # Initialize the spiketrain as below
    mea_map = np.array([[1, 2], [3, 4], [5, 6]])
    mea_map1 = np.array([[0, 0], [0, 0], [0, 0]])
    connectivity_matrix1 = np.array([[2, 3], [10, 11]])
    connectivity_matrix2 = np.array([[2, 3], [5, 8], [9, 10]])

    with pytest.raises(AssertionError):
        plot_connectivity_interactive(mea_map, connectivity_matrix1, False)
    # The function above should throw an error since connectivity matrix is not of same size as mea_map
    with pytest.raises(AssertionError):
        plot_connectivity_interactive(mea_map, connectivity_matrix2, False)
    # The function above should throw an error since connectivity matrix is not a square matrix
    with pytest.raises(AssertionError):
        plot_connectivity_interactive(mea_map1, connectivity_matrix2, True)
        # The function above should throw an error since MEA map contains no identification
