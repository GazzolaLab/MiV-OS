from __future__ import annotations

__doc__ = """
Specification of the behaviors for Operator modules.
"""
__all__ = [
    "OperatorNode",
]

from typing import Protocol, Any
from collections.abc import Callable, Iterator
from typing_extensions import Self

import pathlib
from abc import abstractmethod

from .policy import _RunnerProtocol
from .cachable import _CacherProtocol, CACHE_POLICY
from ..protocol import _Loggable, _Tagged
from miv.core.datatype import DataTypes


class _Runnable(Protocol):
    """
    A protocol for a runner policy.
    """

    @property
    def runner(
        self,
    ) -> _RunnerProtocol: ...

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class _Cachable(_Tagged, _Loggable, _Runnable, Protocol):

    @property
    def cacher(self) -> _CacherProtocol: ...
    @cacher.setter
    def cacher(self, value: _CacherProtocol) -> None: ...

    def set_caching_policy(self, policy: CACHE_POLICY) -> None: ...


class _Chainable(_Cachable, Protocol):
    """
    Behavior includes:
        - Chaining modules in forward/backward linked lists
            - Forward direction defines execution order
            - Backward direction defines dependency order
    """

    _downstream_list: list[_Chainable]
    _upstream_list: list[_Chainable]

    def __rshift__(self, right: _Chainable) -> _Chainable: ...

    def clear_connections(self) -> None: ...

    def summarize(self) -> str:
        """Print summary of downstream network structures."""
        ...

    def _get_upstream_topology(
        self, upstream_nodelist: list[_Chainable] | None = None
    ) -> list[_Chainable]: ...

    def iterate_upstream(self) -> Iterator[_Chainable]: ...

    def iterate_downstream(self) -> Iterator[_Chainable]: ...

    def topological_sort(self) -> list[_Chainable]: ...


class _Callback(Protocol):
    def set_save_path(
        self,
        path: str | pathlib.Path,
        cache_path: str | pathlib.Path | None = None,
    ) -> None: ...

    def __lshift__(self, right: Callable) -> Self: ...

    def reset_callbacks(
        self, *, after_run: bool = False, plot: bool = False
    ) -> None: ...

    def _callback_after_run(self, *args: Any, **kwargs: Any) -> None: ...

    def _callback_plot(
        self,
        output: tuple | None,
        inputs: list | None = None,
        show: bool = False,
        save_path: str | pathlib.Path | None = None,
    ) -> None: ...


class OperatorNode(
    _Callback,
    _Chainable,
    Protocol,
):
    """ """

    analysis_path: str

    def receive(self) -> list[DataTypes]: ...

    def output(self) -> DataTypes: ...

    def run(self) -> DataTypes: ...
