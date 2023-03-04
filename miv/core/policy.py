__all__ = [
    "_Runnable",
    "_RunnerProtocol",
    "VanillaRunner",
    "SupportMultiprocessing",
    "MultiprocessingRunner",
    "InternallyMultiprocessing",
]

from typing import Callable, Optional, Protocol

import multiprocessing
from dataclasses import dataclass


class _RunnerProtocol(Callable, Protocol):
    def __call__(self, func: Callable, inputs: Optional[tuple], **kwargs) -> object:
        ...


class _Runnable(Protocol):
    """
    A protocol for a runner policy.
    """

    @property
    def runner(self) -> _RunnerProtocol:
        ...

    def run(self):
        ...


class VanillaRunner:
    """Default runner without any high-level parallelism."""

    def __call__(self, func, inputs=None, **kwargs):
        if inputs is None:
            output = func()
        else:
            output = func(*inputs)
        return output


class MultiprocessingRunner:
    def __init__(self, np: Optional[int] = None):
        if np is None:
            self._np = multiprocessing.cpu_count()
        else:
            self._np = np

    @property
    def num_proc(self):
        return self._np

    def __call__(self, func, inputs=None, **kwargs):
        if inputs is None:
            raise NotImplementedError(
                "Multiprocessing for operator with no generator input is not supported yet. Please use VanillaRunner for this operator."
            )
        else:
            with multiprocessing.Pool(self.num_proc) as p:
                yield from p.imap(func, inputs)


class StrictMPI:
    pass


class SupportMPI(StrictMPI):
    pass


class SupportMultiprocessing:
    pass


class InternallyMultiprocessing:
    def __init__(self, np: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if np is None:
            self._np = multiprocessing.cpu_count()
        else:
            self._np = np

    @property
    def num_proc(self):
        return self._np
