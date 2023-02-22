__doc__ = """"""
__all__ = ["_Callback"]

from typing import TypeVar  # TODO: For python 3.11, we can use typing.Self
from typing import Callable, Protocol

SelfCallback = TypeVar("SelfCallback", bound="_Callback")


class _Callback(Protocol):
    def __lshift__(self, right: SelfCallback) -> SelfCallback:
        ...

    def callback_before_run(self, **kwargs):
        ...

    def callback_after_run(self, **kwargs):
        ...


class BaseCallbackMixin:
    def __init__(self):
        super().__init__()
        self._callback_before_run = []
        self._callback_after_run = []

    def __lshift__(self, right: Callable) -> SelfCallback:
        if right.__name__.startswith(
            "__prepend"
        ):  # TODO: need better way to prepend callbacks
            self._callback_before_run.append(right)
            return self
        self._callback_after_run.append(right)
        return self

    def callback_before_run(self, inputs):
        for callback in self._callback_before_run:
            callback(self, *inputs)

    def callback_after_run(self, inputs):
        for callback in self._callback_after_run:
            callback(self, *inputs)
