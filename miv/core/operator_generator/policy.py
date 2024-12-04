__all__ = [
    "VanillaGeneratorRunner",
]

from typing import Any, Optional, Protocol, Union
from collections.abc import Callable, Generator

import inspect
import multiprocessing
import pathlib
from dataclasses import dataclass


class VanillaGeneratorRunner:
    """Default runner without any modification.
    Simply, the operator will be executed in embarrassingly parallel manner.
    If MPI is available, the operator will be executed in every ranks.

    This runner is meant to be used for generator operators.
    """

    def __call__(self, func: Callable, inputs: Any | None = None) -> Generator:
        return func(*inputs)
