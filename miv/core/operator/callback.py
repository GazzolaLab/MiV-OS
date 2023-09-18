__doc__ = """"""
__all__ = ["_Callback"]

from typing import TypeVar  # TODO: For python 3.11, we can use typing.Self
from typing import Callable, Optional, Protocol, Union

import inspect
import itertools
import os
import pathlib

import matplotlib.pyplot as plt

SelfCallback = TypeVar("SelfCallback", bound="_Callback")


def MixinOperators(func):
    return func


@MixinOperators
def get_methods_from_feature_classes_by_startswith_str(self, method_name: str):
    methods = [
        [
            v
            for (k, v) in cls.__dict__.items()
            if k.startswith(method_name) and method_name != k and callable(v)
        ]
        for cls in self.__class__.__mro__
    ]
    return list(itertools.chain.from_iterable(methods))


@MixinOperators
def get_methods_from_feature_classes_by_endswith_str(self, method_name: str):
    methods = [
        [
            v
            for (k, v) in cls.__dict__.items()
            if k.endswith(method_name) and method_name != k and callable(v)
        ]
        for cls in self.__class__.__mro__
    ]
    return list(itertools.chain.from_iterable(methods))


class _Callback(Protocol):
    def __lshift__(self, right: SelfCallback) -> SelfCallback:
        ...

    def receive(self):
        ...

    def output(self):
        ...

    def callback_after_run(self):
        ...


class BaseCallbackMixin:
    def __init__(self):
        super().__init__()
        self._callback_collection = []
        self._callback_names = []
        self.skip_plot = False

    def __lshift__(self, right: Callable) -> SelfCallback:
        self._callback_collection.append(right)
        self._callback_names.append(right.__name__)
        return self

    def callback_after_run(self, output):
        predefined_callbacks = get_methods_from_feature_classes_by_startswith_str(
            self, "after_run"
        )
        callback_after_run = []
        for func, name in zip(self._callback_collection, self._callback_names):
            if name.startswith("after_run_"):
                callback_after_run.append(func)

        for callback in predefined_callbacks + callback_after_run:
            output = callback(self, output)

    def plot_from_callbacks(self, *args, **kwargs):
        for func, name in zip(self._callback_collection, self._callback_names):
            if name.startswith("plot_"):
                func(self, *args, **kwargs)

    def plot(
        self,
        output,
        inputs=None,
        show: bool = False,
        save_path: Optional[Union[bool, str, pathlib.Path]] = None,
    ):
        if save_path is True:
            os.makedirs(self.analysis_path, exist_ok=True)
            save_path = self.analysis_path

        plotters = get_methods_from_feature_classes_by_startswith_str(self, "plot_")
        for plotter in plotters:
            plotter(self, output, inputs, show=show, save_path=save_path)
        if not show:
            plt.close("all")
