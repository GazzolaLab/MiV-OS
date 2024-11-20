from __future__ import annotations

__doc__ = """
"""
__all__ = [
    "DefaultLoggerMixin",
]

from typing import TYPE_CHECKING, Any, Generator, Literal, Protocol, Union

import logging
import os
import pathlib
import shutil

if TYPE_CHECKING:
    from miv.core.datatype import DataTypes


class DefaultLoggerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            tag = f"rank[{comm.Get_rank()}]-{self.__class__.__name__}"
        except ImportError:
            tag = self.__class__.__name__
        self.logger = logging.getLogger(tag)

    # TODO: Redirect I/O stream to txt
