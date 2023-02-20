__doc__ = """
"""
__all__ = ["_Cachable", "DataclassCachableMixin"]

from typing import Protocol, Union

import functools
from pathlib import Path


class _Cachable(Protocol):
    @property
    def config_filename(self) -> Union[str, Path]:
        ...

    def load_cache(self) -> bool:
        ...

    def save_cache(self) -> bool:
        ...

    def asdict(self) -> dict:
        ...


class DataclassCachableMixin:
    def load_cache(self) -> bool:
        pass

    def save_cache(self) -> bool:
        pass
