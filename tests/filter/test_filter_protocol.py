from typing import Any, Protocol, Type, runtime_checkable

import pytest

from miv.signal.filter import FilterProtocol
from tests.filter.mock_filter import mock_filter_list, mock_nonfilter_list


@runtime_checkable
class RuntimeFilterProtocol(FilterProtocol, Protocol):
    # This only check the presence of required method, not their type signature.
    # Check @typing.runtime_checkable documentation for more detail.
    ...


@pytest.mark.parametrize("MockFilter", mock_filter_list)
def test_protocol_abide(MockFilter):
    mock_filter = MockFilter()
    assert isinstance(mock_filter, RuntimeFilterProtocol)


@pytest.mark.parametrize("NonFilter", mock_nonfilter_list)
def test_non_protocol_filter(NonFilter):
    none_mock_filter = NonFilter()
    assert not isinstance(none_mock_filter, RuntimeFilterProtocol)
