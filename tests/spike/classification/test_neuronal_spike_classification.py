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


def test_default_model():
    mock_spikes = MockSpikes()
    classifier = NeuronalSpikeClassifier()

    try:
        classifier.create_default_tf_keras_model(
            mock_spikes.spikes[1:], mock_spikes.labels
        )
        assert False
    except Exception:
        pass

    classifier.create_default_tf_keras_model(mock_spikes.spikes, mock_spikes.labels)
    assert np.size(classifier.model.layers) == 3
    classifier.default_compile_model()
    classifier.train_model(x=mock_spikes.spikes, y=mock_spikes.labels, epochs=5)


# def test_get_confusion_matrix():
#     mock_spikes = MockSpikes(size=9)
#     classifier = NeuronalSpikeClassifier()
#     classifier.create_default_tf_keras_model(
#         mock_spikes.spikes, mock_spikes.labels
#     )
#     classifier.default_compile_model()
#     classifier.train_model(
#         mock_spikes.spikes, mock_spikes.labels
#     )
