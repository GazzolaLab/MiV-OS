import os
import sys

import numpy as np
import pytest

from miv.signal.spike.protocol import (
    SpikeFeatureExtractionProtocol,
    UnsupervisedFeatureClusteringProtocol,
)
from miv.signal.spike.sorting import (
    PCADecomposition,
    SpikeSorting,
    WaveletDecomposition,
)

# Test SpikeSorting


class MockDecomposition:
    pass


class MockClustering:
    def fit(self, X):
        pass

    def predict(self, X):
        pass


@pytest.fixture(name="mock_decomposition")
def fixture_create_mock_decomposition() -> SpikeFeatureExtractionProtocol:
    return MockDecomposition()


@pytest.fixture(name="mock_clustering")
def fixture_create_mock_clustering() -> UnsupervisedFeatureClusteringProtocol:
    return MockClustering()


def test_spikesorting_module_init(mock_decomposition, mock_clustering):
    spike_sorting = SpikeSorting(mock_decomposition, mock_clustering)
    assert spike_sorting.featrue_extractor == mock_decomposition
    assert spike_sorting.clustering_method == mock_clustering


@pytest.mark.parametrize("n_group", [-1, 0, 1])
def test_spikesorting_module_n_group_negative_test(
    mock_decomposition, mock_clustering, n_group
):
    spike_sorting = SpikeSorting(mock_decomposition, mock_clustering)
    cutouts = np.ones([10, 100])
    with pytest.raises(AssertionError) as e:
        spike_sorting(cutouts, n_group)
        assert "must be larger than 1" in str(e)


# Test PCA Decomposition
@pytest.mark.parametrize("n_components", [2, 4, 10])
@pytest.mark.parametrize("n_samples", [10, 100, 500])
@pytest.mark.parametrize("cutout_length", [10, 100, 1000])
def test_pca_decomposition_projection_components_shape(
    n_components, n_samples, cutout_length
):
    decomposition = PCADecomposition()
    signal = np.ones([n_samples, cutout_length])
    labels, transformed = decomposition.project(n_components, signal)
    assert labels.shape == (n_samples,)
    assert transformed.shape == (n_samples, 2)
