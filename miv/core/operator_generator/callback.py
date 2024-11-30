__doc__ = """"""

from typing import TypeVar  # TODO: For python 3.11, we can use typing.Self
from typing import Optional, Protocol, Union
from collections.abc import Callable

import inspect
import itertools
import os
import pathlib

import matplotlib.pyplot as plt

from miv.core.operator.callback import (
    BaseCallbackMixin,
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

        self._done_flag_generator_plot = False
        self._done_flag_firstiter_plot = False

    def _reset_callbacks(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._done_flag_generator_plot = getattr(kwargs, "plot", False)
        self._done_flag_firstiter_plot = getattr(kwargs, "plot", False)

    def _callback_generator_plot(
        self,
        iter_index,
        output,
        inputs=None,
        show: bool = False,
        save_path: bool | str | pathlib.Path | None = None,
    ):
        if self._done_flag_generator_plot:
            return

        if save_path is True:
            os.makedirs(self.analysis_path, exist_ok=True)
            save_path = self.analysis_path

        plotters_for_generator_out = get_methods_from_feature_classes_by_startswith_str(
            self, "generator_plot_"
        )
        for plotter in plotters_for_generator_out:
            plotter(
                output,
                inputs,
                show=show,
                save_path=save_path,
                index=iter_index,
            )
        plt.close("all")

    def _callback_firstiter_plot(
        self,
        output,
        inputs=None,
        show: bool = False,
        save_path: bool | str | pathlib.Path | None = None,
    ):
        if self._done_flag_firstiter_plot:
            return

        if save_path is True:
            os.makedirs(self.analysis_path, exist_ok=True)
            save_path = self.analysis_path

        plotters_for_generator_out = get_methods_from_feature_classes_by_startswith_str(
            self, "firstiter_plot_"
        )

        for plotter in plotters_for_generator_out:
            plotter(
                output,
                inputs,
                show=show,
                save_path=save_path,
            )
        plt.close("all")
