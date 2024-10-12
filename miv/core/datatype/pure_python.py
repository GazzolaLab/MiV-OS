__all__ = ["PythonDataType", "NumpyDType", "GeneratorType"]

from typing import Protocol, Union

import numpy as np

from miv.core.operator.chainable import BaseChainingMixin


class RawValuesProtocol(Protocol):
    @staticmethod
    def is_valid(value) -> bool: ...


class ValuesMixin(BaseChainingMixin):
    """
    This mixin is used to convert pure/numpy data type to be a valid input/output of a node.
    """

    def __init__(self, value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = value

    def output(self):
        return self.value

    def run(self, *args, **kwargs):
        return self.output()


class PythonDataType(ValuesMixin):
    """
    Python data types are: int, float, str, bool, list, tuple, dict
    This type also allows None.
    """

    @staticmethod
    def is_valid(value):
        return value is None or isinstance(
            value, (int, float, str, bool, list, tuple, dict)
        )


class NumpyDType(ValuesMixin):
    """
    Numpy data types.
    """

    @staticmethod
    def is_valid(value):
        return isinstance(value, np.ndarray)


class GeneratorType(BaseChainingMixin):
    def __init__(self, iterator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterator = iterator

    def output(self):
        yield from self.iterator

    def run(self, **kwargs):
        yield from self.output()

    @staticmethod
    def is_valid(value):
        import inspect

        return inspect.isgenerator(value)
