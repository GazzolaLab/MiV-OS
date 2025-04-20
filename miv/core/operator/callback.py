__doc__ = """
Implementation for callback features that will be mixed in operator class.
"""
__all__ = ["BaseCallbackMixin"]

from typing import Any, TYPE_CHECKING
from collections.abc import Callable
from typing_extensions import Self

import types
import inspect
import os
import pathlib

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    pass


def MixinOperators(func: Callable) -> Callable:
    return func


@MixinOperators
def get_methods_from_feature_classes_by_startswith_str(
    cls: type, method_name: str
) -> list[Callable]:
    methods = [
        getattr(cls, k)
        for k in dir(cls)
        if k.startswith(method_name) and callable(getattr(cls, k))
    ]
    return methods


@MixinOperators
def get_methods_from_feature_classes_by_endswith_str(
    cls: type, method_name: str
) -> list[Callable]:
    methods = [
        getattr(cls, k)
        for k in dir(cls)
        if k.endswith(method_name) and callable(getattr(cls, k))
    ]
    return methods


class BaseCallbackMixin:
    def __init__(self, cache_path: str = ".cache") -> None:
        super().__init__()
        self.__cache_directory_name: str = cache_path

        # Default analysis path
        assert self.tag != "", (
            "All operator must have self.tag attribute for identification."
        )
        self.set_save_path("results")  # FIXME

        # Callback Flags (to avoid duplicated run)
        self._done_flag_after_run = False
        self._done_flag_plot = False

    def reset_callbacks(self, *, after_run: bool = False, plot: bool = False) -> None:
        self._done_flag_after_run = after_run
        self._done_flag_plot = plot

    def __lshift__(self, right: Callable) -> Self:
        # Dynamically add new function into an operator instance
        if inspect.getfullargspec(right)[0][0] == "self":
            setattr(self, right.__name__, types.MethodType(right, self))
        else:
            # Add new function into as attribute
            setattr(self, right.__name__, right)
        return self

    def set_save_path(
        self,
        path: str | pathlib.Path,
        cache_path: str | pathlib.Path | None = None,
    ) -> None:
        if cache_path is None:
            cache_path = path

        # Set analysis path
        self.analysis_path = os.path.join(path, self.tag.replace(" ", "_"))
        # Set cache path
        _cache_path = os.path.join(
            cache_path, self.tag.replace(" ", "_"), self.__cache_directory_name
        )
        self.cacher.cache_dir = _cache_path

        # Make directory  # Not sure if this needs to be done here
        # os.makedirs(self.analysis_path, exist_ok=True)

    def _callback_after_run(self, *args: Any, **kwargs: Any) -> None:
        if self._done_flag_after_run:
            return

        predefined_callbacks = get_methods_from_feature_classes_by_startswith_str(
            self, "after_run"
        )
        for callback in predefined_callbacks:
            callback(*args, **kwargs)

        self._done_flag_after_run = True

    def _callback_plot(
        self,
        output: Any | None,
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
            plotter(output, inputs, show=show, save_path=save_path)
        if not show:
            plt.close("all")

        self._done_flag_plot = True
