__all__ = ["Extendable"]

from typing import Protocol


class Extendable(Protocol):
    def extend(self, other) -> None:
        ...
