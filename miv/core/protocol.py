from __future__ import annotations

__doc__ = """
This module includes a basic core protocols used in many place throughout.
the library. For specific behaviors, protocols will be specified in the
module/protocol.py files.
"""
__all__ = ["_Loggable"]

from collections.abc import Iterator
from typing import Any, Protocol, TypeVar
import logging

from .cachable import _CacherProtocol, CACHE_POLICY


class _Cachable(Protocol):
    """
    A protocol for cachable behavior.
    """

    cacher: _CacherProtocol

    def set_caching_policy(self, policy: CACHE_POLICY) -> None: ...


class _Loggable(Protocol):
    """
    A protocol for a logger policy.
    """

    @property
    def logger(self) -> logging.Logger: ...


class _Jsonable(Protocol):
    def to_json(self) -> dict[str, Any]: ...

    # TODO: need more features to switch the I/O of the logger or MPI-aware logging.


C = TypeVar("C", bound="_Chainable")


class _Chainable(Protocol[C]):
    """
    Defines the behavior for chaining operator modules:
    - Forward direction defines execution order
    - Backward direction defines dependency order
    """

    def __rshift__(self, right: C) -> C: ...

    # def append_upstream(self, node: C) -> None: ...

    # def append_downstream(self, node: C) -> None: ...

    # def disconnect_upstream(self, node: C) -> None: ...

    # def disconnect_downstream(self, node: C) -> None: ...

    def iterate_upstream(self) -> Iterator[C]: ...

    def iterate_downstream(self) -> Iterator[C]: ...

    def flow_blocked(self) -> bool: ...
