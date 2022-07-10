from typing import List

from unittest import mock

import numpy as np

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
        AbnormalityDetector.num_components = num_components
        AbnormalityDetector.model = None
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
    abn_detector = MockAbnormalDetector(chan_spike_cutouts, 3, 6)

    cat_list = [[2, 1, 0], [2, 1, 0]]
    abn_detector.categorize_spontaneous(cat_list)

    assert abn_detector.categorized
    for chan_index, chan_spike_cutout in enumerate(abn_detector.spontaneous_cutouts):
        for cutout_index, elem in enumerate(chan_spike_cutout.categorization_list):
            assert elem == cat_list[chan_index][cutout_index]
