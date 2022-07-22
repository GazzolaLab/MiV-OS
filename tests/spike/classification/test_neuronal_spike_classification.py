import numpy as np

from miv.signal.classification.neuronal_spike_classification import (
    NeuronalSpikeClassifier,
)
from tests.spike.cutout.test_cutout import MockSpikeCutout


class MockSpikes:
    def __init__(self) -> None:
        spikes = []
        labels = []
        for i in range(3):
            spikes.append(MockSpikeCutout(i, i, i * 0.1).cutout)
            labels.append(i)
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
