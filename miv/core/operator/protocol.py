__doc__ = """
Specification of the behaviors for Operator modules.
"""

from typing import Protocol, Any, TypeVar
from collections.abc import Iterator

import pathlib
from abc import abstractmethod

from .policy import _RunnerProtocol
from .cachable import _CacherProtocol, CACHE_POLICY
from ..protocol import _Loggable
from miv.core.datatype import DataTypes

C = TypeVar("C", bound="_Chainable")


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


class _Cachable(Protocol):
    """
    A protocol for cachable behavior.
    """

    cacher: _CacherProtocol

    def set_caching_policy(self, policy: CACHE_POLICY) -> None: ...


class _Chainable(Protocol[C]):
    """
    Defines the behavior for chaining operator modules:
    - Forward direction defines execution order
    - Backward direction defines dependency order
    """

    def append_upstream(self, node: C) -> None: ...

    def append_downstream(self, node: C) -> None: ...

    def disconnect_upstream(self, node: C) -> None: ...

    def disconnect_downstream(self, node: C) -> None: ...

    def iterate_upstream(self) -> Iterator[C]: ...

    def iterate_downstream(self) -> Iterator[C]: ...


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
