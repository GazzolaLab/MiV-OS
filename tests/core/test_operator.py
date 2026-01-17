"""
Tests for OperatorMixin.
"""

import pytest

from miv.core.operator.operator import OperatorMixin


def test_operator_mixin_cannot_be_directly_instantiated():
    """
    OperatorMixin is supposed to be used as a mixin for another class, and not directly instantiated.
    """
    with pytest.raises(TypeError):
        OperatorMixin()


def test_operator_mixin_can_be_used_as_mixin():
    """
    OperatorMixin should work correctly when used as a mixin in a subclass.
    """

    class MockOperator(OperatorMixin):
        tag: str = "test operator"

        def __call__(self):
            return "result"

    # Should be able to instantiate a subclass
    operator = MockOperator()
    assert operator.tag == "test operator"
    assert hasattr(operator, "runner")
    assert hasattr(operator, "cacher")
