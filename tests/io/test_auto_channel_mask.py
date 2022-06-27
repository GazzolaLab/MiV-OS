import unittest
from contextlib import contextmanager
from unittest import mock
from unittest.mock import call, create_autospec

import numpy as np
import pytest

from miv.io.data import DataManager
from miv.signal.filter import ButterBandpass
from miv.signal.spike import ThresholdCutoff
from tests.io.mock_data import MockData, MockDataManager


class TestAutoChannelMask(unittest.TestCase):
    def test_auto_channel_mask_baseline(self):
        band_filter = ButterBandpass(300, 3000)
        detector = ThresholdCutoff()
        data_man = MockDataManager(None)
        data_man.auto_channel_mask_baseline(band_filter, detector)

        assert data_man[0].masking_channel_set == {0, 1, 2, 3}
        assert data_man[1].masking_channel_set == {0, 1, 2, 3}

    def test_clear_channel_masks(self):
        band_filter = ButterBandpass(300, 3000)
        detector = ThresholdCutoff()
        data_man = MockDataManager(None)
        data_man.auto_channel_mask_baseline(band_filter, detector)

        data_man[0].clear_channel_mask()
        data_man[1].clear_channel_mask()
        assert len(data_man[0].masking_channel_set) == 0
        assert len(data_man[1].masking_channel_set) == 0

    def test_get_binned_matrix_empty_channels(self):
        band_filter = ButterBandpass(300, 3000)
        detector = ThresholdCutoff()
        data_man = MockDataManager(None)
        data_man.auto_channel_mask_baseline(band_filter, detector)

        assert data_man[0]._get_binned_matrix(band_filter, detector)[
            "empty_channels"
        ] == [0, 1, 2, 3]
        assert data_man[1]._get_binned_matrix(band_filter, detector)[
            "empty_channels"
        ] == [0, 1, 2, 3]

    # def test_auto_channel_mask(self):
    #     band_filter = ButterBandpass(300, 3000)
    #     detector = ThresholdCutoff()
    #     data_man = MockDataManager(None)

    #     data_man.auto_channel_mask(data_man[0], band_filter, detector)
    #     assert data_man[0].masking_channel_set == {0, 1, 2, 3}
