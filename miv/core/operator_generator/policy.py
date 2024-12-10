__all__ = [
    "VanillaGeneratorRunner",
]

from typing import Any, Optional, Protocol, Union
from collections.abc import Callable, Generator, Sequence

import inspect
import multiprocessing
import pathlib
from dataclasses import dataclass

from ..protocol import _LazyCallable


class VanillaGeneratorRunner:
    """Default runner without any modification.
    Simply, the operator will be executed in embarrassingly parallel manner.
    If MPI is available, the operator will be executed in every ranks.

    This runner is meant to be used for generator operators.
    """

    def __call__(
        self, func: _LazyCallable, inputs: list[Generator[Any]] | None = None
    ) -> Generator:
        # FIXME: fix type
        return func(*inputs)  # type: ignore

    def get_run_order(self) -> int:
        return 0
