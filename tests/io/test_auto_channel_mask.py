import unittest
from contextlib import contextmanager
from unittest import mock
from unittest.mock import call, create_autospec

import numpy as np
import pytest

from miv.signal.filter import ButterBandpass
from miv.signal.spike import ThresholdCutoff
from tests.io.mock_data import MockData, MockDataManager, MockSpontaneousData


def test_auto_channel_mask_with_firing_rate():
    band_filter = ButterBandpass(300, 3000)
    detector = ThresholdCutoff()
    data_man = MockDataManager(None)
    data_man.auto_channel_mask_with_firing_rate(band_filter, detector)

    assert data_man[0].masking_channel_set == {0, 1, 2, 3}
    assert data_man[1].masking_channel_set == {0, 1, 2, 3}


def test_clear_channel_masks():
    band_filter = ButterBandpass(300, 3000)
    detector = ThresholdCutoff()
    data_man = MockDataManager(None)
    data_man.auto_channel_mask_with_firing_rate(band_filter, detector)

    data_man[0].clear_channel_mask()
    assert len(data_man[0].masking_channel_set) == 0
    assert data_man[1].masking_channel_set == {0, 1, 2, 3}


def test_get_binned_matrix_empty_channels():
    band_filter = ButterBandpass(300, 3000)
    detector = ThresholdCutoff()
    data_man = MockDataManager(None)
    data_man.auto_channel_mask_with_firing_rate(band_filter, detector)

    assert data_man[0]._get_binned_matrix(band_filter, detector)["empty_channels"] == [
        0,
        1,
    ]
    assert data_man[1]._get_binned_matrix(band_filter, detector)["empty_channels"] == [
        0,
        1,
    ]


def test_get_binned_matrix():
    band_filter = ButterBandpass(300, 3000)
    detector = ThresholdCutoff()
    data_man = MockDataManager(None)

    binned = data_man[0]._get_binned_matrix(band_filter, detector, bins_per_second=2)
    assert np.any(np.transpose(binned["matrix"])[5], where=3)


def test_auto_channel_mask_with_correlation_matrix():
    band_filter = ButterBandpass(300, 3000)
    detector = ThresholdCutoff()
    data_man = MockDataManager(None)

    spontaneous_data = MockSpontaneousData()
    data_man.auto_channel_mask_with_correlation_matrix(
        spontaneous_data, band_filter, detector
    )

    # channels 1 and 2 should be masked because they have no spikes
    # channel 2 and 3 should be masked as their spikes are itendical to the spontaneous data
    assert data_man[0].masking_channel_set == {0, 1, 2, 3}
