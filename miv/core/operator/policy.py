"""
This module contains the runner policies for the operator system.
"""

from __future__ import annotations

__all__ = [
    "RunnerBase",
    "VanillaRunner",
    "StrictMPIRunner",
    "SupportMPIMerge",
]
from typing import TYPE_CHECKING, cast
from collections.abc import Callable

from ..policy import RunnerBase

from miv.import_helper import require_library
from miv.core.datatype.operation.concatenate import concatenate

if TYPE_CHECKING:
    # This will likely cause circular import error
    from miv.core.datatype import DataTypes
    import mpi4py


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


@require_library(["mpi4py"])
class StrictMPIRunner(RunnerBase):
    def __init__(self, *, comm: mpi4py.MPI.Comm | None = None, root: int = 0) -> None:
        if comm is not None:
            self.comm = comm
        else:
            from mpi4py import MPI

            self.comm = MPI.COMM_WORLD
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
        output = super().__call__(func, inputs)

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
        output = super().__call__(func, inputs)

        outputs = self.comm.gather(output, root=self.root)
        if self.is_root():
            result = concatenate(outputs)  # Class method
        else:
            result = None
        return result
