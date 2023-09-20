import networkx as nx
import numpy as np
import pytest
from mock_connectivity import MockConnectivity

from miv.statistics.connectivity import plot_eigenvector_centrality

# Import the function to be tested here


class TestPlotEigenvectorCentrality:
    @pytest.fixture
    def result(self):
        # Create a sample result object to be used in the tests
        connectivity_matrix = np.array(
            [[0.0, 1.0, 0.5], [1.0, 0.0, 0.3], [0.5, 0.3, 0.0]]
        )
        adjacency_matrix = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        return {
            "connectivity_matrix": connectivity_matrix,
            "adjacency_matrix": adjacency_matrix,
        }

    def test_plot_run(self, result):
        # Test the plot_centrality function
        plot_eigenvector_centrality(MockConnectivity(), result, inputs=None)
