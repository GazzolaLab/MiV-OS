from typing import runtime_checkable
from typing import Protocol, Any

import pytest

import numpy as np

from miv.signal.filter import FilterProtocol, FilterCollection
from tests.filter.mock_filter import mock_filter_list, mock_nonfilter_list

from tests.filter.test_filter_protocol import RuntimeFilterProtocol


def test_empty_filter_protocol_abide():
    empty_filter = FilterCollection()
    assert isinstance(empty_filter, RuntimeFilterProtocol)


@pytest.mark.parametrize("MockFilter1", mock_filter_list)
@pytest.mark.parametrize("MockFilter2", mock_filter_list)
def test_mock_filter_collection_protocol_abide(
    MockFilter1: FilterProtocol, MockFilter2: FilterProtocol
):
    filter_collection = FilterCollection().append(MockFilter1()).append(MockFilter2())
    assert isinstance(filter_collection, RuntimeFilterProtocol)

    filter_collection = FilterCollection().append(MockFilter1()).append(MockFilter1())
    assert isinstance(filter_collection, RuntimeFilterProtocol)

    filter_collection = (
        FilterCollection().append(MockFilter1()).insert(0, MockFilter1())
    )
    assert isinstance(filter_collection, RuntimeFilterProtocol)

    filter_collection = (
        FilterCollection().insert(0, MockFilter1()).insert(0, MockFilter1())
    )
    assert isinstance(filter_collection, RuntimeFilterProtocol)


def test_filter_collection_representation():
    filter_collection = FilterCollection()
    assert "Collection" in repr(filter_collection)

    tag = "whats the point of doing a PhD?"
    filter_collection = FilterCollection(tag=tag)
    assert tag in repr(filter_collection)


class TestFilterCollectionMutableSequence:
    @pytest.fixture(scope="class")
    def load_collection(self):
        flt = FilterCollection()
        # Bypass check, but its fine for testing
        flt.append(3)
        flt.append(5.0)
        flt.append("a")
        flt.append(np.random.randn(3, 5))
        return flt

    def test_len(self, load_collection):
        assert len(load_collection) == 4

    def test_getitem(self, load_collection):
        assert load_collection[0] == 3
        assert load_collection[2] == "a"

    @pytest.mark.xfail
    def test_getitem_with_faulty_index_fails(self, load_collection):
        # Fails and exception is raised
        load_collection[100]

    @pytest.mark.xfail
    def test_setitem_with_faulty_index_fails(self, load_collection):
        # If this fails, an exception is raised
        # and pytest automatically fails
        load_collection[200] = 1.0

    def test_setitem(self, load_collection):
        # If this fails, an exception is raised
        # and pytest automatically fails
        load_collection[3] = 1.0

    def test_insert(self, load_collection):
        load_collection.insert(3, "ss")
        assert load_collection[3] == "ss"
        load_collection.insert(1, 1.0)
        assert np.isclose(load_collection[1], 1.0)
        assert load_collection[4] == "ss"

    def test_str(self, load_collection):
        assert str(load_collection[0]) == "3"

    @pytest.mark.xfail
    def test_delitem(self, load_collection):
        del load_collection[0]
        assert load_collection[0] == 3


class TestFilterCollectionFunctionality:
    @pytest.mark.parametrize("sampling_rate", [20, 50, 0])
    def test_empty_filter_collection_bypass(self, sampling_rate):
        flt = FilterCollection()
        test_signal = np.random.random([2, 50])
        filtered_signal = flt(test_signal, sampling_rate)
        np.testing.assert_allclose(test_signal, filtered_signal)

    @pytest.fixture(scope="function")
    def mock_filter(self):
        from tests.filter import mock_filter_list

        MockFilter = type("MockFilter", (mock_filter_list[0],), {})
        return MockFilter()


class TestFilterCollectionCompatibility:
    """
    Collection of compatibility test of FilterCollection with other filter modules
    """

    def test_filter_collection_with_butterworth_io_shape(self):
        from miv.signal.filter import ButterBandpass

        flt = (
            FilterCollection()
            .append(ButterBandpass(1, 2))
            .append(ButterBandpass(2, 3))
            .append(ButterBandpass(3, 4))
        )
        sig = np.random.random([2, 50])
        filt_sig = flt(sig, sampling_rate=1000)
        np.testing.assert_equal(sig.shape, filt_sig.shape)

    from tests.filter.test_butterworth import (
        AnalyticalTestSet as ButterworthAnalyticalTestSet,
    )
    from tests.filter.test_butterworth import ParameterSet as ButterworthParameterSet

    @pytest.mark.parametrize("lowcut, highcut, order, tag", ButterworthParameterSet[:1])
    @pytest.mark.parametrize("sig, rate, result", ButterworthAnalyticalTestSet)
    def test_filter_collection_with_butterworth_value(
        self, lowcut, highcut, order, tag, sig, rate, result
    ):
        from miv.signal.filter import ButterBandpass

        filt = FilterCollection().append(ButterBandpass(lowcut, highcut, order, tag))
        ans = filt(signal=sig, sampling_rate=rate)
        np.testing.assert_allclose(ans, result)

    @pytest.mark.parametrize("lowcut, highcut, order, tag", ButterworthParameterSet)
    def test_filter_collection_repr_string_with_butterworth(
        self, lowcut, highcut, order, tag
    ):
        from miv.signal.filter import ButterBandpass

        filt = FilterCollection().append(ButterBandpass(lowcut, highcut, order, tag))
        for v in [lowcut, highcut, order, tag]:
            assert str(v) in repr(filt)


class TestFilterCollectionIntegration:
    """
    Collection of integration test of FilterCollection with other modules
    """

    def test_filter_collection_operate_on_dataset(self):
        pass  # TODO
