__doc__ = """"""

from typing import TypeVar  # TODO: For python 3.11, we can use typing.Self
from typing import Callable, Optional, Protocol, Union

import inspect
import itertools
import os
import pathlib

import matplotlib.pyplot as plt

from miv.core.operator.callback import (
    BaseCallbackMixin,
    SelfCallback,
    get_methods_from_feature_classes_by_startswith_str,
)


class GeneratorCallbackMixin:
    """
    Additional methods for generator-to-generator operator.

    `generator_plot` method plots during each iteration of generator.
    The function take `show` and `save_path` arguments similar to `plot` method.
    """

    def __init__(self):
        super().__init__()

    def generator_plot_from_callbacks(self, *args, **kwargs):
        for func, name in zip(self._callback_collection, self._callback_names):
            if name.startswith("generator_plot_"):
                func(self, *args, **kwargs)

    def generator_plot(
        self,
        iter_index,
        output,
        inputs=None,
        show: bool = False,
        save_path: Optional[Union[bool, str, pathlib.Path]] = None,
    ):
        if save_path is True:
            os.makedirs(self.analysis_path, exist_ok=True)
            save_path = self.analysis_path

        plotters_for_generator_out = get_methods_from_feature_classes_by_startswith_str(
            self, "generator_plot_"
        )
        for plotter in plotters_for_generator_out:
            plotter(
                self,
                output,
                inputs,
                show=show,
                save_path=save_path,
                index=iter_index,
            )
        if not show:
            plt.close("all")

    def firstiter_plot_from_callbacks(self, *args, **kwargs):
        for func, name in zip(self._callback_collection, self._callback_names):
            if name.startswith("firstiter_plot_"):
                func(self, *args, **kwargs)

    def firstiter_plot(
        self,
        output,
        inputs=None,
        show: bool = False,
        save_path: Optional[Union[bool, str, pathlib.Path]] = None,
    ):
        if save_path is True:
            os.makedirs(self.analysis_path, exist_ok=True)
            save_path = self.analysis_path

        plotters_for_generator_out = get_methods_from_feature_classes_by_startswith_str(
            self, "firstiter_plot_"
        )

        for plotter in plotters_for_generator_out:
            plotter(
                self,
                output,
                inputs,
                show=show,
                save_path=save_path,
            )
        if not show:
            plt.close("all")
