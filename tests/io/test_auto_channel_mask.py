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
    @mock.patch("miv.io.data")
    def test_spike_detection_was_called_in_baseline(self, mock_object):
        band_filter = ButterBandpass(300, 3000)
        detector = mock_object()
        mock_data = mock_object()

        @contextmanager
        def mock_load_funct():
            times = np.arange(start=0, stop=1, step=1 / 10000.0)
            signal = np.transpose(np.zeros(10000))
            yield signal, times, 10000.0

        mock_load = create_autospec(mock_load_funct)
        mock_data.load.return_value = mock_load()

        manager = MockDataManager(mock_data)
        manager.auto_channel_mask_baseline(band_filter, detector)
        detector.assert_has_called()

    # # @mock.patch("miv.io.data")
    # def test_auto_channel_mask_baseline(self):
    #     band_filter = ButterBandpass(300, 3000)
    #     detector = ThresholdCutoff()
    #     data_man = MockDataManager()

    #     data_man.auto_channel_mask_baseline(band_filter, detector)
    #     with data_man[0].load() as (sig, times, samp):
    #         print(sig)
    #         print("sig shape", np.shape(sig))
    #         assert(np.shape(sig)[0] == 0)

    # @mock.patch("miv.io.data")
    # def test_get_binned_matrix(self, mock_data):

    #     band_filter = ButterBandpass(300, 3000)
    #     detector = ThresholdCutoff()

    #     mock_data.load.return_value = ([], [], 0)
    #     mock_data._get_binned_matrix.return_value = {
    #         "matrix": [[]],
    #         "num_bins": [],
    #         "empty_channels": [],
    #     }
    #     mock_data._auto_channel_mask.side_effect = mock_data._get_binned_matrix(
    #         band_filter, detector
    #     )

    #     data_manager = MockDataManager(mock_data)
    #     data_manager.auto_channel_mask(mock_data(), band_filter, detector)

    #     calls = [
    #         call(band_filter, detector),
    #         call(band_filter, detector),
    #         call(band_filter, detector),
    #     ]

    #     mock_data._get_binned_matrix.assert_has_calls(calls)
