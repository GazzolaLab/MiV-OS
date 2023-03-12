from typing import Generator, Protocol

from miv.core.datatype.protocol import Extendable


class _Collapsable(Protocol):
    @classmethod
    def from_collapse(self) -> None:
        ...


class CollapseExtendableMixin:
    def __init__(self):
        super().__init__()

    @classmethod
    def from_collapse(cls, values: Generator[Extendable, None, None]):
        obj = cls()
        for value in values:
            obj.extend(value)
        return obj