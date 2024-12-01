__doc__ = """Protocol for Data objects."""
__all__ = ["DataProtocol"]

import typing
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)
from collections.abc import Callable, Generator, Iterable

import os

import numpy as np
from miv.typing import SignalType


class DataProtocol(Protocol):
    """Behavior definition for a single experimental data handler."""

    def __init__(self, data_path: str): ...

    @property
    def analysis_path(self) -> None: ...

    def load(self, *args) -> Generator[SignalType, np.ndarray, int]:
        """Iterator to load data fragmentally. Use to load large file size data."""
        ...

    def check_path_validity(self) -> bool:
        """Check if necessary files exist in the directory."""
        ...
