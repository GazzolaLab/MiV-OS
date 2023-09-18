__all__ = [
    "VanillaGeneratorRunner",
]

from typing import Any, Callable, Generator, Optional, Protocol, Union

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

    def __init__(self):
        pass

    def __call__(self, func, inputs, **kwargs):
        return func(*inputs)
