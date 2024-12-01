__all__ = [
    "_Runnable",
    "_RunnerProtocol",
    "VanillaRunner",
    "SupportMultiprocessing",
    "MultiprocessingRunner",
    "InternallyMultiprocessing",
    "StrictMPIRunner",
    "SupportMPIMerge",
]

import inspect
import multiprocessing
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, Union
from collections.abc import Callable, Generator

if TYPE_CHECKING:
    # This will likely cause circular import error
    from miv.core.datatype.collapsable import _Collapsable


class _RunnerProtocol(Callable, Protocol):
    def __init__(self, comm, root: int): ...

    def __call__(self, func: Callable, inputs: tuple | None, **kwargs) -> object: ...


class _Runnable(Protocol):
    """
    A protocol for a runner policy.
    """

    @property
    def runner(self) -> _RunnerProtocol: ...

    def run(
        self,
        save_path: str | pathlib.Path,
        cache_dir: str | pathlib.Path,
    ) -> None: ...


class VanillaRunner:
    """Default runner without any high-level parallelism.
    Simply, the operator will be executed in root-rank, and distributed across other ranks.
    If MPI is not available, the operator will be executed in root-rank only.
    """

    def __init__(self):
        try:
            from mpi4py import MPI

            self.comm = MPI.COMM_WORLD
            self.is_root = self.comm.Get_rank() == 0
            if self.comm.Get_size() == 1:
                self.comm = None
                self.is_root = True
        except ImportError:
            self.comm = None
            self.is_root = True

    def _execute(self, func, inputs):
        if inputs is None:
            output = func()
        else:
            output = func(*inputs)
        return output

    def __call__(self, func, inputs=None, **kwargs):
        output = None
        if self.is_root:
            output = self._execute(func, inputs)

        if self.comm is not None:
            output = self.comm.bcast(output, root=0)
        return output


class MultiprocessingRunner:
    def __init__(self, np: int | None = None):
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
    def __init__(self, comm=None, root=0):
        if comm is not None:
            self.comm = comm
        else:
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


class SupportMPIMerge(StrictMPIRunner):
    """
    This runner policy is used for operators that can be merged by MPI.
    """

    def __call__(self, func, inputs=None, **kwargs):
        if inputs is None:
            output: _Collapsable = func()
        else:
            output: _Collapsable = func(*inputs)

        outputs = self.comm.gather(output, root=self.root)
        if self.is_root():
            result = output.from_collapse(outputs)  # Class method
        else:
            result = None
        result = self.comm.bcast(result, root=self.root)
        return result

class SupportMPIWithoutBroadcast(StrictMPIRunner):
    """
    This runner policy is used for operators that can be merged by MPI.
    """

    def __call__(self, func, inputs=None, **kwargs):
        if inputs is None:
            output: _Collapsable = func()
        else:
            output: _Collapsable = func(*inputs)

        outputs = self.comm.gather(output, root=self.root)
        if self.is_root():
            result = output.from_collapse(outputs)  # Class method
        else:
            result = None
        return result


class SupportMPI(StrictMPIRunner):
    pass


class SupportMultiprocessing:
    pass


class InternallyMultiprocessing(Protocol):
    @property
    def num_proc(self) -> int: ...
