__all__ = ["PythonDataType", "NumpyDType", "GeneratorType"]

from typing import TypeAlias, Any
from collections.abc import Generator, Iterator

import numpy as np

from .node_mixin import DataNodeMixin
from ..chainable import ChainingMixin

PurePythonTypes: TypeAlias = int | float | str | bool | list | tuple | dict


class ValuesMixin(DataNodeMixin, ChainingMixin):
    """
    This mixin is used to convert pure/numpy data type to be a valid input/output of a node.
    """

    def __init__(
        self, data: np.ndarray | PurePythonTypes, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.data = data


class PythonDataType(ValuesMixin):
    """
    Python data types are: int, float, str, bool, list, tuple, dict
    This type also allows None.
    """

    @staticmethod
    def is_valid(data: Any) -> bool:
        return data is None or isinstance(
            data, int | float | str | bool | list | tuple | dict
        )


class NumpyDType(ValuesMixin):
    """
    Numpy data types.
    """

    @staticmethod
    def is_valid(data: Any) -> bool:
        return isinstance(data, np.ndarray)


class GeneratorType(ChainingMixin):
    def __init__(self, iterator: Iterator, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.iterator = iterator

    def flow_blocked(self) -> bool:
        return False

    def output(self) -> Generator:
        yield from self.iterator

    def run(self, **kwargs: Any) -> Generator:
        yield from self.output()

    @staticmethod
    def is_valid(data: Any) -> bool:
        import inspect

        return inspect.isgenerator(data)
