from typing import Any
from collections.abc import Iterable

import pytest

from miv.core.datatype.mixin_colapsable import ConcatenateMixin


class MockExtendable(ConcatenateMixin):
    def __init__(self, value: Any | None = None) -> None:
        if value is None:
            self._value = []
        else:
            self._value = [value]

    @property
    def value(self) -> list[Any]:
        return self._value

    def __eq__(self, other: Any) -> bool:
        """Used for comparison"""
        if isinstance(other, list):
            return self.value == other
        return False

    def extend(self, other: "MockExtendable") -> "MockExtendable":
        self._value.extend(other.value)
        return self


class TestCollapseExtendableMixin:
    def test_concatenate(self) -> None:
        values = [MockExtendable(1), MockExtendable(2), MockExtendable(3)]
        obj = MockExtendable.concatenate(values)
        assert obj.value == [1, 2, 3]

    def test_concatenate_single_item(self) -> None:
        """Test that concatenate works with a single item."""
        values = [MockExtendable(1)]
        obj = MockExtendable.concatenate(values)
        assert obj.value == [1]

    def test_concatenate_empty_list(self) -> None:
        """Test that concatenate can work with an empty list when a head is provided."""
        head = MockExtendable(0)
        obj = MockExtendable.concatenate([], head=head)
        assert obj.value == [0]

    def test_concatenate_no_values(self) -> None:
        """Test that concatenate raises an exception with an empty list and no head."""
        with pytest.raises(
            TypeError
        ):  # Expects error when reducing empty sequence with no initial value
            MockExtendable.concatenate([])

    def test_return_type_same(self) -> None:
        """Test that the return type of concatenate is the same as the class it's called on."""
        values = [MockExtendable(1), MockExtendable(2)]
        obj = MockExtendable.concatenate(values)
        assert isinstance(obj, MockExtendable)

        # Test with head parameter too
        head = MockExtendable(0)
        obj_with_head = MockExtendable.concatenate(values, head=head)
        assert isinstance(obj_with_head, MockExtendable)

    def test_order_preservation(self) -> None:
        """Test that the order of concatenation is preserved."""
        empty = MockExtendable()
        values = [MockExtendable("a"), MockExtendable("b"), MockExtendable("c")]
        obj = MockExtendable.concatenate(values, head=empty)
        assert obj.value == ["a", "b", "c"]

    def test_original_inplace_extend(self) -> None:
        """Test that original objects are not modified during concatenation."""
        a = MockExtendable(1)
        b = MockExtendable(2)
        values = [a, b]
        obj = MockExtendable.concatenate(values)

        assert obj.value == [1, 2]
        assert a.value == [1, 2]
        assert b.value == [2]

    def test_original_not_modified(self) -> None:
        """Test that original objects are not modified during concatenation."""
        empty = MockExtendable()
        a = MockExtendable(1)
        b = MockExtendable(2)
        values = [a, b]
        obj = MockExtendable.concatenate(values, head=empty)

        assert obj.value == [1, 2]
        assert a.value == [1]
        assert b.value == [2]

    def test_chained_concatenation(self) -> None:
        """Test that calling concatenate multiple times works correctly."""
        first_set = [MockExtendable(1), MockExtendable(2)]
        second_set = [MockExtendable(3), MockExtendable(4)]

        # Firt result2.value == [1, 2, 3, 4]
        obj = MockExtendable.concatenate(first_set)
        obj = MockExtendable.concatenate(second_set, head=obj)
        assert obj.value == [1, 2, 3, 4]
