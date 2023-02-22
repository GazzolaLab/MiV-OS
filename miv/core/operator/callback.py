__doc__ = """"""
__all__ = ["_Callback"]

from typing import TypeVar  # TODO: For python 3.11, we can use typing.Self
from typing import Callable, Protocol

SelfCallback = TypeVar("SelfCallback", bound="_Callback")


class _Callback(Protocol):
    def __lshift__(self, right: SelfCallback) -> SelfCallback:
        ...

    def receive(self):
        ...

    def output(self):
        ...

    def callback_before_run(self):
        ...

    def callback_after_run(self):
        ...


class BaseCallbackMixin:
    def __init__(self):
        super().__init__()
        self._callback_before_run = []
        self._callback_after_run = []
        self._callback_plot = []

    def __lshift__(self, right: Callable) -> SelfCallback:
        if right.__name__.startswith(
            "__prepend"
        ):  # TODO: need better way to prepend callbacks
            self._callback_before_run.append(right)
            return self
        if right.__name__.startswith(
            "plot_"
        ):  # TODO: need better way to prepend callbacks
            self._callback_plot.append(right)
            return self
        self._callback_after_run.append(right)
        return self

    def callback_before_run(self):
        for callback in self._callback_before_run:
            callback(self, self.receive())

    def callback_after_run(self):
        for callback in self._callback_after_run:
            callback(self, self.output)

    def plot_from_callbacks(self, *args, **kwargs):
        for callback in self._callback_plot:
            callback(self, *args, **kwargs)
