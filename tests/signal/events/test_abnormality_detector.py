from typing import List

from unittest import mock

import numpy as np
import tensorflow as tf

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
    assert abn_detector.model.layers[0].trainable


# def test_train_model():
#     chan_spike_cutouts = []
#     for i in range(2):
#         cutouts = []
#         cutouts.append(MockSpikeCutout(0, 0, 0))
#         cutouts.append(MockSpikeCutout(1, 1, 0.1))
#         cutouts.append(MockSpikeCutout(2, 2, 0.2))
#         cutouts.append(MockSpikeCutout(0, 0, 0.3))
#         cutouts.append(MockSpikeCutout(1, 1, 0.4))
#         cutouts.append(MockSpikeCutout(2, 2, 0.5))
#         chan_spike_cutouts.append(ChannelSpikeCutout(cutouts, 3, 0))
#     abn_detector = MockAbnormalDetector(chan_spike_cutouts, 3, 2)

#     abn_detector.categorize_spontaneous([[-1, 1, 0], [-1, 1, 0]])
#     abn_detector.train_model()

#     test_cutout0 = np.array(MockSpikeCutout(0, 0, 0.6).cutout)
#     test_cutout1 = np.array(MockSpikeCutout(1, 1, 0.7).cutout)
#     test_cutout2 = np.array(MockSpikeCutout(2, 2, 0.8).cutout)

#     prob_model = tf.keras.Sequential([abn_detector.model, tf.keras.layers.Softmax()])

#     prediction0 = prob_model.predict(test_cutout0, verbose=0)
#     prediction1 = prob_model.predict(test_cutout1, verbose=0)
#     prediction2 = prob_model.predict(test_cutout2, verbose=0)

#     assert np.argmax(prediction0) == -1
#     assert np.argmax(prediction1) == 1
#     assert np.argmax(prediction2) == 0
