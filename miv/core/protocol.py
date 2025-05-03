from __future__ import annotations

__doc__ = """
This module includes a basic core protocols used in many place throughout.
the library. For specific behaviors, protocols will be specified in the
module/protocol.py files.
"""
__all__ = ["_Loggable"]

from collections.abc import Callable, Generator
from typing import Any, Protocol
import logging

# Lazy-callable function takes generators as input and returns a generator
_LazyCallable = Callable[[Generator[Any]], Generator[Any]]  # FIXME


class _Loggable(Protocol):
    """
    A protocol for a logger policy.
    """

    @property
    def logger(self) -> logging.Logger: ...


class _Jsonable(Protocol):
    def to_json(self) -> dict[str, Any]: ...

    # TODO: need more features to switch the I/O of the logger or MPI-aware logging.
