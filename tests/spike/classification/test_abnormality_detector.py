from typing import List

from unittest import mock

import numpy as np
import tensorflow as tf
from grpc import Channel

from miv.signal.events.abnormality_detection import AbnormalityDetector
from miv.signal.spike.cutout import ChannelSpikeCutout, SpikeCutout
from tests.spike.test_cutout import MockSpikeCutout


class MockAbnormalDetector(AbnormalityDetector):
    def __init__(
        self,
        spont_cutouts: List[ChannelSpikeCutout],
        num_components: int,
        num_channels: int,
    ):
        AbnormalityDetector.num_channels = num_channels
        AbnormalityDetector.trained = False
        AbnormalityDetector.extractor_decomposition_parameter = num_components
        AbnormalityDetector.categorized = False
        AbnormalityDetector.spontaneous_cutouts = spont_cutouts

    def _get_cutouts(self):
        return super().spontaneous_cutouts


def test_categorize_spontaneous():
    chan_spike_cutouts = []
    for i in range(2):
        cutouts = []
        cutouts.append(MockSpikeCutout(0, 0, 0))
        cutouts.append(MockSpikeCutout(1, 1, 0.1))
        cutouts.append(MockSpikeCutout(2, 2, 0.2))
        cutouts.append(MockSpikeCutout(0, 0, 0.3))
        cutouts.append(MockSpikeCutout(1, 1, 0.4))
        cutouts.append(MockSpikeCutout(2, 2, 0.5))
        chan_spike_cutouts.append(ChannelSpikeCutout(cutouts, 3, 0))
    abn_detector = MockAbnormalDetector(chan_spike_cutouts, 3, 2)

    cat_list = [[2, 1, 0], [2, 1, 0]]
    abn_detector.categorize_spontaneous(cat_list)

    assert abn_detector.categorized
    for chan_index, chan_spike_cutout in enumerate(abn_detector.spontaneous_cutouts):
        for cutout_index, elem in enumerate(chan_spike_cutout.categorization_list):
            assert elem == cat_list[chan_index][cutout_index]


def test_create_default_model():
    # 2 identical channels with 6 cutouts each
    chan_spike_cutouts = []
    for i in range(2):
        cutouts = []
        cutouts.append(MockSpikeCutout(0, 0, 0, 40))
        cutouts.append(MockSpikeCutout(1, 1, 0.1, 40))
        cutouts.append(MockSpikeCutout(2, 2, 0.2, 40))
        cutouts.append(MockSpikeCutout(0, 0, 0.3, 40))
        cutouts.append(MockSpikeCutout(1, 1, 0.4, 40))
        cutouts.append(MockSpikeCutout(2, 2, 0.5, 40))
        chan_spike_cutouts.append(ChannelSpikeCutout(cutouts, 3, 0))
    abn_detector = MockAbnormalDetector(chan_spike_cutouts, 3, 2)

    abn_detector._create_default_model(40, 80, 2)

    assert abn_detector.model is not None
    assert len(abn_detector.model.layers) == 3
    for layer_index, layer in enumerate(abn_detector.model.layers):
        assert layer.trainable


def test_evaluate_model():
    train_cutouts = np.ndarray((6, 40))
    train_cutouts[0] = MockSpikeCutout(0, 0, 0.0, 40).cutout
    train_cutouts[1] = MockSpikeCutout(1, 1, 0.1, 40).cutout
    train_cutouts[2] = MockSpikeCutout(2, 2, 0.2, 40).cutout
    train_cutouts[3] = MockSpikeCutout(0, 0, 0.3, 40).cutout
    train_cutouts[4] = MockSpikeCutout(1, 1, 0.4, 40).cutout
    train_cutouts[5] = MockSpikeCutout(2, 2, 0.5, 40).cutout
    train_labels = np.array([0, 1, 2, 0, 1, 2])

    test_cutouts = train_cutouts[0:4]
    test_labels = np.array([0, 1, 2, 1])

    layers = [
        tf.keras.layers.Dense(40),
        tf.keras.layers.Dense(120),
        tf.keras.layers.Dense(3),
    ]
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    # overfitting on purpose
    model.fit(train_cutouts, train_labels, epochs=40, verbose=0)

    chan_spike_cutouts = []
    for i in range(2):
        cutouts = []
        cutouts.append(MockSpikeCutout(0, 0, 0, 40))
        cutouts.append(MockSpikeCutout(1, 1, 0.1, 40))
        cutouts.append(MockSpikeCutout(2, 2, 0.2, 40))
        cutouts.append(MockSpikeCutout(0, 0, 0.3, 40))
        cutouts.append(MockSpikeCutout(1, 1, 0.4, 40))
        cutouts.append(MockSpikeCutout(2, 2, 0.5, 40))
        chan_spike_cutouts.append(ChannelSpikeCutout(cutouts, 3, 0))
    abn_detector = MockAbnormalDetector(chan_spike_cutouts, 3, 2)

    abn_detector.model = model
    abn_detector.trained = False
    assert abn_detector.evaluate_model(test_cutouts, test_labels)["test_accuracy"] == 0
    abn_detector.trained = True
    assert (
        abn_detector.evaluate_model(test_cutouts, test_labels)["test_accuracy"] == 0.75
    )


def test_train_model():
    cutouts = []
    cutouts.append(MockSpikeCutout(0, 0, 0))
    cutouts.append(MockSpikeCutout(1, 1, 0.1))
    cutouts.append(MockSpikeCutout(2, 2, 0.2))
    cutouts.append(MockSpikeCutout(0, 0, 0.3))
    cutouts.append(MockSpikeCutout(1, 1, 0.4))
    cutouts.append(MockSpikeCutout(2, 2, 0.5))
    cat_list = [[0, 1, -1]]
    chan0_spike_cutout = ChannelSpikeCutout(cutouts, 3, 0)
    abn_detector = MockAbnormalDetector([chan0_spike_cutout], 3, 1)

    abn_detector.categorize_spontaneous(cat_list)

    abn_detector.train_model(epochs=10)

    test0 = np.array(MockSpikeCutout(0, 0, 0).cutout)
    # test1 = np.array(MockSpikeCutout(1, 0, 0).cutout)

    assert np.shape(test0) == np.shape(cutouts[0].cutout)

    # prob_model = tf.keras.Sequential([abn_detector.model, tf.keras.layers.Softmax()])
    # prob_model = abn_detector.model

    # pred0 = prob_model.predict(test0, verbose=0)
    # pred1 = prob_model.predict(test1, verbose=0)
