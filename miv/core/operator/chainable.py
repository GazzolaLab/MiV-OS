__doc__ = """"""
__all__ = ["_Chainable", "BaseChainingMixin"]

from typing import TypeVar  # TODO: For python 3.11, we can use typing.Self
from typing import Callable, Iterator, List, Optional, Protocol, Union

import functools

from miv.typing import DataTypes

SelfChain = TypeVar("SelfChain", bound="_Chainable")


class _Chainable(Protocol):
    """
    Behavior includes:
        - Chaining modules in forward/backward linked lists
            - Forward direction defines execution order
            - Backward direction defines dependency order
    """

    def __rshift__(self, right: SelfChain) -> SelfChain:
        ...

    def iterate_upstream(self) -> Iterator[SelfChain]:
        ...

    def iterate_downstream(self) -> Iterator[SelfChain]:
        ...


class BaseChainingMixin:
    """
    Base mixin to create chaining structure between objects.
    """

    def __init__(self):
        super().__init__()
        self._downstream_list: List[_Chainable] = []
        self._upstream_list: List[_Chainable] = []

        self._output: Optional[DataTypes] = None
        self._flag_executed = False

    def __rshift__(self, right: _Chainable) -> _Chainable:
        self._downstream_list.append(right)
        right._upstream_list.append(self)
        return self

    def iterate_downstream(self) -> Iterator[_Chainable]:
        return iter(self._downstream_list)

    def iterate_upstream(self) -> Iterator[_Chainable]:
        return iter(self._upstream_list)
