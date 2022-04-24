from typing import runtime_checkable
from typing import Protocol, Any

import pytest

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
