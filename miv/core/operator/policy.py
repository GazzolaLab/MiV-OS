"""
This module contains the runner policies for the operator system.
"""

from __future__ import annotations

__all__ = [
    "RunnerBase",
    "VanillaRunner",
    "SupportMultiprocessing",
    "InternallyMultiprocessing",
    "StrictMPIRunner",
    "SupportMPIMerge",
]
from typing import TYPE_CHECKING, Any, cast
from collections.abc import Callable, Generator
from abc import ABC, abstractmethod

from miv.core.datatype.operation.concatenate import concatenate

if TYPE_CHECKING:
    # This will likely cause circular import error
    from miv.core.datatype import DataTypes
    import mpi4py


class RunnerBase(ABC):
    def __call__(
        self, func: Callable, inputs: Any | None = None
    ) -> Generator[Any] | Any: ...

    @abstractmethod
    def get_run_order(self) -> int:
        """
        The method determines the order of execution, useful for
        multiprocessing or MPI.
        """


class VanillaRunner(RunnerBase):
    """
    Default runner without any high-level parallelism.
    Simply, the operator will be executed in root-rank, and distributed across other ranks.
    """

    def __init__(self, *, comm: mpi4py.MPI.Comm | None = None, root: int = 0) -> None:
        self.comm: mpi4py.MPI.Comm | None
        self.is_root: bool

        if comm is not None:
            self.comm = comm
            self.is_root = self.comm.Get_rank() == root
        else:
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

    def get_run_order(self) -> int:
        return 0

    def _execute(self, func: Callable, inputs: DataTypes) -> DataTypes:
        if inputs is None:
            output = func()
        else:
            inputs = cast("DataTypes", inputs)
            output = func(*inputs)
        return output

    def __call__(self, func: Callable, inputs: DataTypes | None = None) -> DataTypes:
        output = None
        if self.is_root:
            output = self._execute(func, inputs)

        if self.comm is not None:
            output = self.comm.bcast(output, root=0)
        return output


class StrictMPIRunner(RunnerBase):
    def __init__(self, *, comm: mpi4py.MPI.Comm, root: int = 0) -> None:
        self.comm = comm
        self.root = root

    def get_run_order(self) -> int:
        return int(self.get_rank())

    def get_rank(self) -> int:
        return int(self.comm.Get_rank())

    def get_size(self) -> int:
        return int(self.comm.Get_size())

    def get_root(self) -> int:
        return self.root

    def is_root(self) -> bool:
        return self.get_rank() == self.root

    def __call__(self, func: Callable, inputs: DataTypes | None = None) -> DataTypes:
        if inputs is None:
            output = func()
        else:
            inputs = cast(tuple["DataTypes"], inputs)
            output = func(*inputs)
        return output


class SupportMPIMerge(StrictMPIRunner):
    """
    This runner policy is used for operators that can be merged by MPI.
    """

    def __call__(self, func: Callable, inputs: DataTypes | None = None) -> DataTypes:
        if inputs is None:
            output = func()
        else:
            inputs = cast(tuple["DataTypes"], inputs)
            output = func(*inputs)

        outputs = self.comm.gather(output, root=self.root)
        if self.is_root():
            result = concatenate(outputs)  # Class method
        else:
            result = None
        result = self.comm.bcast(result, root=self.root)
        return result


class SupportMPIWithoutBroadcast(StrictMPIRunner):
    """
    This runner policy is used for operators that can be merged by MPI.
    """

    def __call__(
        self, func: Callable, inputs: tuple[DataTypes] | None = None
    ) -> DataTypes | None:
        if inputs is None:
            output = func()
        else:
            output = func(*inputs)

        outputs = self.comm.gather(output, root=self.root)
        if self.is_root():
            result = concatenate(outputs)  # Class method
        else:
            result = None
        return result


class SupportMultiprocessing:
    pass


class InternallyMultiprocessing:
    @property
    def num_proc(self) -> int: ...
