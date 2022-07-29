import numpy as np

from miv.signal.classification.neuronal_spike_classification import (
    NeuronalSpikeClassifier,
)
from tests.spike.cutout.test_cutout import MockSpikeCutout


class MockSpikes:
    def __init__(self, size: int = 3, length=40) -> None:
        spikes = []
        labels = []
        self.size = size
        for i in range(self.size):
            spikes.append(MockSpikeCutout(i % 3, i % 3, i * 0.1, length=length).cutout)
            labels.append(i % 3)
        self.spikes = np.array(spikes)
        self.labels = np.array(labels)


def test_create_default_model():
    mock_spikes = MockSpikes(size=3, length=40)
    classifier = NeuronalSpikeClassifier()

    classifier.create_default_tf_keras_model(40)
    assert np.size(classifier.model.layers) == 3
    classifier.default_compile_model()
    classifier.train_model(x=mock_spikes.spikes, y=mock_spikes.labels, epochs=5)


class DistinctiveMockSpikes:
    def __init__(self, size: int = 40):
        spike0 = np.ones(size)
        spike1 = np.linspace(start=-10, stop=10, num=size)
        self.spikes = np.array([spike0, spike1, spike0, spike1])
        self.labels = np.array([0, 1, 0, 1])


def test_predict_categories_sigmoid():
    mock_spikes = DistinctiveMockSpikes(size=30)
    classifier = NeuronalSpikeClassifier()
    classifier.create_default_tf_keras_model(30)
    classifier.default_compile_model()
    classifier.train_model(
        x=mock_spikes.spikes,
        y=mock_spikes.labels,
        epochs=100,  # Just 4 samples per epoch so this should be fine
    )

    predictions = classifier.predict_categories_sigmoid(mock_spikes.spikes)
    for spike_index, spike_prediction in enumerate(predictions):
        assert mock_spikes.labels[spike_index] == spike_prediction
