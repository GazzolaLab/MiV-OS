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
    mea_map_t = np.array([[1, 2]])
    connectivity_matrix1 = np.array([[2, 3], [10, 11]])
    connectivity_matrix2 = np.array([[2, 3], [5, 8], [9, 10]])
    connectivity_matrix_t = np.array([[0, 0.1], [0.1, 0]])

    with pytest.raises(AssertionError):
        plot_connectivity(mea_map, connectivity_matrix1, False)
    # The function above should throw an error since connectivity matrix is not of same size as mea_map
    with pytest.raises(AssertionError):
        plot_connectivity(mea_map, connectivity_matrix2, False)
    # The function above should throw an error since connectivity matrix is not a square matrix
    with pytest.raises(AssertionError):
        plot_connectivity(mea_map1, connectivity_matrix2, True)
        # The function above should throw an error since MEA map contains no identification
    output = plot_connectivity(mea_map_t, connectivity_matrix_t, False)
    assert output[0].engine == "neato"
    output = plot_connectivity(mea_map_t, connectivity_matrix_t, True)
    assert output[0].engine == "neato"


def test_plot_connectivity_interactive_output():

    # Initialize the spiketrain as below
    mea_map = np.array([[1, 2], [3, 4], [5, 6]])
    mea_map1 = np.array([[0, 0], [0, 0], [0, 0]])
    mea_map_t = np.array([[1, 2]])
    connectivity_matrix1 = np.array([[2, 3], [10, 11]])
    connectivity_matrix2 = np.array([[2, 3], [5, 8], [9, 10]])
    connectivity_matrix_t = np.array([[0, 0.1], [0.1, 0]])

    with pytest.raises(AssertionError):
        plot_connectivity_interactive(mea_map, connectivity_matrix1, False)
    # The function above should throw an error since connectivity matrix is not of same size as mea_map
    with pytest.raises(AssertionError):
        plot_connectivity_interactive(mea_map, connectivity_matrix2, False)
    # The function above should throw an error since connectivity matrix is not a square matrix
    with pytest.raises(AssertionError):
        plot_connectivity_interactive(mea_map1, connectivity_matrix2, True)
        # The function above should throw an error since MEA map contains no identification

    output = plot_connectivity_interactive(mea_map_t, connectivity_matrix_t, False)
    assert output[0].get_edges()[0]["width"] == "1"
    output = plot_connectivity_interactive(mea_map_t, connectivity_matrix_t, True)
    assert output[0].get_edges()[0]["width"] == "1"
