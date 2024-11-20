__doc__ = """
Implementation for callback features that will be mixed in operator class.
"""
__all__ = ["BaseCallbackMixin"]

from typing import TypeVar  # TODO: For python 3.11, we can use typing.Self
from typing import Callable, Optional, Protocol, Union

import types
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
        getattr(self, k)
        for k in dir(self)
        if k.startswith(method_name) and callable(getattr(self, k))
    ]
    return methods


@MixinOperators
def get_methods_from_feature_classes_by_endswith_str(self, method_name: str):
    methods = [
        getattr(self, k)
        for k in dir(self)
        if k.endswith(method_name) and callable(getattr(self, k))
    ]
    return methods


class BaseCallbackMixin:
    def __init__(self, cache_path: str = ".cache"):
        super().__init__()
        self.__cache_directory_name: str = cache_path

        # Default analysis path
        assert (
            self.tag != ""
        ), "All operator must have self.tag attribute for identification."
        self.set_save_path("results")

        # Callback Flags (to avoid duplicated run)
        self._done_flag_after_run = False
        self._done_flag_plot = False

    def _reset_callbacks(self, *, after_run: bool = False, plot: bool = False) -> None:
        self._done_flag_after_run = after_run
        self._done_flag_plot = plot

    def __lshift__(self, right: Callable) -> SelfCallback:
        # Dynamically add new function into an operator instance
        if inspect.getfullargspec(right)[0][0] == "self":
            setattr(self, right.__name__, types.MethodType(right, self))
        else:
            # Add new function into as attribute
            setattr(self, right.__name__, right)
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
        _cache_path = os.path.join(
            cache_path, self.tag.replace(" ", "_"), self.__cache_directory_name
        )
        self.cacher.cache_dir = _cache_path

        # Make directory
        os.makedirs(self.analysis_path, exist_ok=True)

    def _callback_after_run(self, *args, **kwargs):
        if self._done_flag_after_run:
            return

        predefined_callbacks = get_methods_from_feature_classes_by_startswith_str(
            self, "after_run"
        )
        for callback in predefined_callbacks:
            callback(self, *args, **kwargs)

        self._done_flag_after_run = True

    def _callback_plot(
        self,
        output: tuple | None,
        inputs: list | None = None,
        show: bool = False,
        save_path: str | pathlib.Path | None = None,
    ) -> None:
        """
        Run all function in this operator that starts with the name 'plot_'.
        """
        if self._done_flag_plot:
            return

        if save_path is None:
            save_path = self.analysis_path

        # If input is single-argument, strip the list
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]

        plotters = get_methods_from_feature_classes_by_startswith_str(self, "plot_")
        for plotter in plotters:
            plotter(self, output, inputs, show=show, save_path=save_path)
        if not show:
            plt.close("all")

        self._done_flag_plot = True
