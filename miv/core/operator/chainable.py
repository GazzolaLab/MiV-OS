__doc__ = """"""
__all__ = ["_Chainable", "BaseChainingMixin"]

from typing import TypeVar  # TODO: For python 3.11, we can use typing.Self
from typing import Callable, Iterator, List, Optional, Protocol, Union

import functools

from miv.core.datatype import DataTypes

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

    def summarize(self) -> str:
        """Print summary of downstream network structures."""
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

    def __rshift__(self, right: _Chainable) -> _Chainable:
        self._downstream_list.append(right)
        right._upstream_list.append(self)
        return right

    def iterate_downstream(self) -> Iterator[_Chainable]:
        return iter(self._downstream_list)

    def iterate_upstream(self) -> Iterator[_Chainable]:
        return iter(self._upstream_list)

    def summarize(self) -> str:  # TODO: create DFS and BFS traverser
        q = [(0, self)]
        order = []  # DFS
        while len(q) > 0:
            depth, current = q.pop(0)
            if current in order:  # Avoid loop
                continue
            q += [(depth + 1, node) for node in current.iterate_downstream()]
            order.append((depth, current))
        return self._text_visualize_hierarchy(order)

    def _text_visualize_hierarchy(self, string_list, prefix="|__ "):
        output = []
        for i, item in enumerate(string_list):
            depth, label = item
            if depth > 0:
                output.append("    " * (depth - 1) + prefix + str(label))
            else:
                output.append(str(label))
        return "\n".join(output)
