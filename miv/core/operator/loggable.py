from __future__ import annotations

__doc__ = """
"""
__all__ = [
    "_Loggable",
    "DefaultLoggerMixin",
]

from typing import TYPE_CHECKING, Any, Generator, Literal, Protocol, Union

import logging
import os
import pathlib
import shutil

if TYPE_CHECKING:
    from miv.core.datatype import DataTypes


class _Loggable(Protocol):
    @property
    def logger(self):
        ...


class DefaultLoggerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: account for multiprocessing and MPI
        self.logger = logging.getLogger(self.__class__.__name__)
