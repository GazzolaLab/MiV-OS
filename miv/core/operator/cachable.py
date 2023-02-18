__doc__ = """"""
__all__ = ["_Cachable"]

from typing import Protocol

import functools


class _Cachable(Protocol):
    def load_cache(self) -> bool:
        ...

    def save_cache(self) -> bool:
        ...
