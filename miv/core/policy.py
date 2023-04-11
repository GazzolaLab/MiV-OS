__all__ = [
    "_Runnable",
    "_RunnerProtocol",
    "VanillaRunner",
    "SupportMultiprocessing",
    "MultiprocessingRunner",
    "InternallyMultiprocessing",
    "StrictMPIRunner",
]

from typing import Any, Callable, Generator, Optional, Protocol, Union

import multiprocessing
import pathlib
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

    def run(
        self,
        save_path: Union[str, pathlib.Path],
        dry_run: bool,
        cache_dir: Union[str, pathlib.Path],
    ) -> None:
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

    def __call__(self, func, inputs: Generator[Any, None, None] = None, **kwargs):
        if inputs is None:
            raise NotImplementedError(
                "Multiprocessing for operator with no generator input is not supported yet. Please use VanillaRunner for this operator."
            )
        else:
            with multiprocessing.Pool(self.num_proc) as p:
                yield from p.imap(func, inputs)


class StrictMPIRunner:
    def __init__(self):
        from mpi4py import MPI

        self.comm = MPI.COMM_WORLD
        self.root = 0

    def set_comm(self, comm):
        self.comm = comm

    def set_root(self, root: int):
        self.root = root

    def get_rank(self):
        return self.comm.Get_rank()

    def get_size(self):
        return self.comm.Get_size()

    def get_root(self):
        return self.root

    def is_root(self):
        return self.get_rank() == self.root

    def __call__(self, func, inputs=None, **kwargs):
        if inputs is None:
            output = func()
        else:
            output = func(*inputs)
        return output


class SupportMPI(StrictMPIRunner):
    pass


class SupportMultiprocessing:
    pass


class InternallyMultiprocessing(Protocol):
    @property
    def num_proc(self) -> int:
        ...
