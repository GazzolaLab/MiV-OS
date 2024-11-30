__doc__ = """"""
__all__ = ["_Callback"]

from typing import TypeVar  # TODO: For python 3.11, we can use typing.Self
from typing import Callable, Optional, Protocol, Union
from typing_extensions import Self

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
    def __lshift__(self, right: Callable) -> Self: ...

    def receive(self): ...

    def output(self): ...

    def callback_after_run(self): ...

    def set_save_path(self, path: Union[str, pathlib.Path]) -> None: ...

    def make_analysis_path(self) -> None: ...


class BaseCallbackMixin:
    def __init__(self):
        super().__init__()
        self._callback_collection = []
        self._callback_names = []
        self.skip_plot = False

    def __lshift__(self, right: Callable) -> Self:
        self._callback_collection.append(right)
        self._callback_names.append(right.__name__)
        return self

    def set_save_path(
        self,
        path: Union[str, pathlib.Path],
        cache_path: Union[str, pathlib.Path] = None,
    ):
        if cache_path is None:
            cache_path = path

        # Set analysis path
        self.analysis_path = os.path.join(path, self.tag.replace(" ", "_"))
        # Set cache path
        _cache_path = os.path.join(cache_path, self.tag.replace(" ", "_"), ".cache")
        self.cacher.cache_dir = _cache_path

    def make_analysis_path(self):
        os.makedirs(self.analysis_path, exist_ok=True)

    def callback_after_run(self, *args, **kwargs):
        predefined_callbacks = get_methods_from_feature_classes_by_startswith_str(
            self, "after_run"
        )
        callback_after_run = []
        for func, name in zip(self._callback_collection, self._callback_names):
            if name.startswith("after_run_"):
                callback_after_run.append(func)

        for callback in predefined_callbacks + callback_after_run:
            callback(self, *args, **kwargs)

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
        # TODO: Not sure if excluding none-output is a good idea
        if output is None:
            return
        if save_path is True:
            os.makedirs(self.analysis_path, exist_ok=True)
            save_path = self.analysis_path

        plotters = get_methods_from_feature_classes_by_startswith_str(self, "plot_")
        for plotter in plotters:
            plotter(self, output, inputs, show=show, save_path=save_path)
        if not show:
            plt.close("all")
