__doc__ = """Protocol for Data objects."""
__all__ = ["DataProtocol"]

import typing
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

import os

from miv.typing import SignalType, TimestampsType


class DataProtocol(Protocol):
    """Behavior definition for a single experimental data handler."""

    def __init__(self, data_path: str):
        ...

    @property
    def analysis_path(self) -> None:
        ...

    def load(
        self, start_at_zero: bool = False
    ) -> Tuple[SignalType, TimestampsType, int]:
        """Context manager for loading data instantly."""
        ...

    def load_fragments(
        self, start_at_zero: bool, num_fragments: int, progress_bar=False
    ) -> Generator[SignalType, TimestampsType, int]:
        """Iterator to load data fragmentally. Use to load large file size data."""
        ...

    def check_path_validity(self) -> bool:
        """Check if necessary files exist in the directory."""
        ...
