from __future__ import annotations

__doc__ = """
"""
__all__ = [
    "DefaultLoggerMixin",
]

from typing import TYPE_CHECKING, Any

import logging

if TYPE_CHECKING:
    pass


class DefaultLoggerMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            if comm.Get_size() > 1:
                tag = f"rank[{comm.Get_rank()}]-{self.__class__.__name__}"
            else:
                tag = self.__class__.__name__
        except ImportError:
            tag = self.__class__.__name__
        self._logger = logging.getLogger(tag)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    # TODO: Redirect I/O stream to txt
