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

import inspect
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
    """Default runner without any high-level parallelism.

    If MPI is available, only use first rank (root) to execute, and other ranks recv from the root.

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
            # TODO: support kwargs
            output = self._execute(func, inputs)

        # If MPI is available:
        if self.comm is not None:  # MPI # FIXME
            # If output is generator, other ranks also need to initialize generator.
            # Otherwise, broadcast output
            is_generator_output = None
            if self.is_root:
                is_generator_output = inspect.isgenerator(output)
            is_generator_output = self.comm.bcast(is_generator_output, root=0)

            if is_generator_output:
                if not self.is_root:
                    output = self._execute(func, inputs)
            else:
                output = self.comm.bcast(output, root=0)
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
