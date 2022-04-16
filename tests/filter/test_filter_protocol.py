from typing import runtime_checkable
from typing import Protocol, Any

import pytest

from miv.signal.filter import FilterProtocol
from tests.filter.mock_filter import mock_filter_list, mock_nonfilter_list


@runtime_checkable
class RuntimeFilterProtocol(FilterProtocol, Protocol):
    # This only check the presence of required method, not their type signature.
    # Check @typing.runtime_checkable documentation for more detail.
    ...


@pytest.mark.parametrize("MockFilter", mock_filter_list)
def test_protocol_abide(MockFilter: FilterProtocol):
    mock_filter: FilterProtocol = MockFilter()
    assert isinstance(mock_filter, RuntimeFilterProtocol)


@pytest.mark.parametrize("NonFilter", mock_nonfilter_list)
def test_non_protocol_filter(NonFilter: Any):
    none_mock_filter: Any = NonFilter()
    with pytest.raises(TypeError):
        issubclass(none_mock_filter, RuntimeFilterProtocol)
