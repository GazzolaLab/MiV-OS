__doc__ = """
Specification of the behaviors for Operator modules.
"""

from typing import Protocol, Any
from abc import abstractmethod

from .policy import RunnerBase

__all__ = ["_Runnable"]


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
