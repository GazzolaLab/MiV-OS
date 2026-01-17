__doc__ = """
Specification of the behaviors for Operator modules.
"""

from typing import Protocol, Any

import pathlib
from abc import abstractmethod

from .policy import RunnerBase
from ..protocol import _Loggable, _Chainable, _Cachable
from ..datatype import DataTypes


class _Runnable(Protocol):
    """
    A protocol for a runner policy.
    """

    @property
    def runner(
        self,
    ) -> RunnerBase: ...

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class _Node(
    _Loggable,
    _Runnable,
    _Chainable,
    _Cachable,
    Protocol,
):
    """
    Protocol defining the complete behavior of an Operator node.
    Each protocol aspect is separately defined to allow for cleaner composition.
    """

    analysis_path: str

    def output(self) -> DataTypes: ...
    def set_save_path(
        self,
        path: str | pathlib.Path,
        cache_path: str | pathlib.Path | None = None,
    ) -> None: ...
