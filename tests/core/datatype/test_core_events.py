import numpy as np
import pytest
import quantities as pq

from miv.core import Events, Signal


class TestEvents:
    @pytest.fixture
    def events(self):
        return Events([0.1, 0.5, 1.2, 1.5, 2.3, 3.0])

    def test_len(self, events):
        assert len(events) == 6

    def test_get_last_event(self, events):
        assert events.get_last_event() == 3.0

    def test_get_first_event(self, events):
        assert events.get_first_event() == 0.1

    def test_get_view(self, events):
        view = events.get_view(1.0, 2.0)
        np.testing.assert_array_equal(view.data, [1.2, 1.5])

    def test_binning(self, events):
        signal = events.binning(bin_size=0.5)
        assert isinstance(signal, Signal)
        assert signal.shape == (6, 1)
        assert signal.rate == 2.0

        signal = events.binning(bin_size=1.0 * pq.ms, return_count=True)
        assert signal.shape == (2900, 1)
        assert signal.rate == 1000.0

    def test_binning_invalid_bin_size(self, events):
        with pytest.raises(AssertionError):
            events.binning(bin_size=-0.5)

    def test_binning_with_units(self, events):
        signal = events.binning(bin_size=2 * pq.ms)
        assert signal.shape == (1450, 1)
        assert signal.rate == 500.0


def test_negative_bin_size():
    events = Events([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(AssertionError):
        events.binning(bin_size=-1 * pq.ms)


def test_empty_events_binning():
    events = Events()
    with pytest.raises(ValueError):
        events.binning(bin_size=1 * pq.ms)


def test_invalid_t_start():
    events = Events([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        events.binning(bin_size=1 * pq.ms, t_start=0.5)


def test_invalid_t_end():
    events = Events([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        events.binning(bin_size=1 * pq.ms, t_end=0.05)


def test_return_count_binning():
    events = Events([0.1, 0.2, 0.3, 0.4])
    signal = events.binning(bin_size=1 * pq.ms, return_count=True)
    assert signal.data.shape == (301, 1)
    assert signal.data[0, 0] == 1
