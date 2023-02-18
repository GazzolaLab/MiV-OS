__doc__ = """"""
__all__ = ["_Callback"]

from typing import TypeVar  # TODO: For python 3.11, we can use typing.Self
from typing import Protocol

SelfCallback = TypeVar("SelfCallback", bound="_Callback")


class _Callback(Protocol):
    def __lshift__(self, right: SelfCallback) -> SelfCallback:
        ...

    def before_run(self, **kwargs):
        ...

    def after_run(self, **kwargs):
        ...
