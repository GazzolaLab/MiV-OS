import unittest
from unittest.mock import call, mock

import pytest
from mock_data import MockData, MockDataManager

from miv.io.data import DataManager
from miv.signal.filter import ButterBandpass
from miv.signal.spike import ThresholdCutoff


class TestAutoChannelMask(unittest.TestCase):
    @mock.patch("miv.io.data")
    def test_get_binned_matrix(self, mock_data):

        band_filter = ButterBandpass(300, 3000)
        detector = ThresholdCutoff()

        mock_data.load.return_value = ([], [], 0)
        mock_data._get_binned_matrix.return_value = {
            "matrix": [[]],
            "num_bins": [],
            "empty_channels": [],
        }
        mock_data._auto_channel_mask.side_effect = mock_data._get_binned_matrix(
            band_filter, detector
        )

        data_manager = MockDataManager(mock_data)
        data_manager.auto_channel_mask(mock_data(), band_filter, detector)

        calls = [
            call(band_filter, detector),
            call(band_filter, detector),
            call(band_filter, detector),
        ]

        mock_data._get_binned_matrix.assert_has_calls(calls)
