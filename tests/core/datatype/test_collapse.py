from typing import Generator

import pytest

from miv.core.datatype.collapsable import CollapseExtendableMixin


class MockExtendable(CollapseExtendableMixin):
    def __init__(self, value=None):
        if value is None:
            self._value = []
        else:
            self._value = [value]

    @property
    def value(self):
        return self._value

    def __eq__(self, other):
        return self.value == other

    def extend(self, other):
        return self._value.extend(other.value)


class TestCollapseExtendableMixin:
    def test_from_collapse(self):
        values = [MockExtendable(1), MockExtendable(2), MockExtendable(3)]
        obj = MockExtendable.from_collapse(values)
        assert obj == [1, 2, 3]
