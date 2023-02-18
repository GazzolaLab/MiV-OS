__doc__ = """"""
__all__ = ["_Chainable", "BaseChainingMixin"]

from typing import TypeVar  # TODO: For python 3.11, we can use typing.Self
from typing import Callable, Iterator, List, Protocol, Union

import functools

SelfChain = TypeVar("SelfChain", bound="_Chainable")


class _Chainable(Protocol):
    def __rshift__(self, right: SelfChain) -> SelfChain:
        ...

    def iterate_next(self) -> Iterator[SelfChain]:
        ...


class BaseChainingMixin:
    def __init__(self):
        super().__init__()
        self._next_list: List[_Chainable] = []

    def __rshift__(self, right: _Chainable) -> _Chainable:
        self._next_list.append(right)
        return self

    def iterate_next(self) -> Iterator[_Chainable]:
        return iter(self._next_list)
