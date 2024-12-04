from typing import Any, Protocol
from collections.abc import Iterable


class _Collapsable(Protocol):
    @classmethod
    def from_collapse(self, values: Iterable["_Collapsable"]) -> "_Collapsable": ...

    def extend(self, *args: Any, **kwargs: Any) -> None: ...


class CollapseExtendableMixin:
    @classmethod
    def from_collapse(cls, values: Iterable[_Collapsable]) -> _Collapsable:
        obj: _Collapsable
        for idx, value in enumerate(values):
            if idx == 0:
                obj = value
            else:
                obj.extend(value)
        return obj
