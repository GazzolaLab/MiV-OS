"""Callback mixin for generator operators.

This module provides callback functionality for generator-to-generator operators,
including plotting callbacks that execute during generator iterations.
"""

from typing import TYPE_CHECKING, Any, Protocol

import pathlib

import matplotlib.pyplot as plt

from ..loggable import DefaultLoggerMixin
from ..operator.callback import (
    get_methods_from_feature_classes_by_startswith_str,
    execute_callback,
)

if TYPE_CHECKING:
    from miv.core.datatype import DataTypes


class _GeneratorCallback(Protocol):
    _done_flag_generator_plot: bool
    _done_flag_firstiter_plot: bool

    def _callback_generator_plot(
        self,
        iter_index: int,
        output: "DataTypes",
        inputs: tuple["DataTypes", ...] | None = None,
        show: bool = False,
        save_path: str | pathlib.Path | None = None,
    ) -> None: ...

    def _callback_firstiter_plot(
        self,
        output: "DataTypes",
        inputs: tuple["DataTypes", ...] | None = None,
        show: bool = False,
        save_path: str | pathlib.Path | None = None,
    ) -> None: ...


class GeneratorCallbackMixin(DefaultLoggerMixin):
    """
    Additional methods for generator-to-generator operator.

    `generator_plot` method plots during each iteration of generator.
    The function take `show` and `save_path` arguments similar to `plot` method.
    """

    def __init__(self) -> None:
        super().__init__()

        self._done_flag_generator_plot = False
        self._done_flag_firstiter_plot = False

    def reset_callbacks(self, *args: Any, **kwargs: Any) -> None:
        """Reset callback flags.

        Args:
            *args: Positional arguments (unused, for compatibility).
            **kwargs: Keyword arguments. If 'plot' is provided, sets both
                generator_plot and firstiter_plot flags to that value.
        """
        plot_flag = kwargs.get("plot", False)
        self._done_flag_generator_plot = plot_flag
        self._done_flag_firstiter_plot = plot_flag

    def _callback_generator_plot(
        self,
        iter_index: int,
        output: "DataTypes",
        inputs: tuple["DataTypes", ...] | None = None,
        show: bool = False,
        save_path: str | pathlib.Path | None = None,
    ) -> None:
        if self._done_flag_generator_plot:
            return

        plotters_for_generator_out = get_methods_from_feature_classes_by_startswith_str(
            self, "generator_plot_"
        )
        for plotter in plotters_for_generator_out:
            execute_callback(
                self.logger,
                plotter,
                output,
                inputs,
                show=show,
                save_path=save_path,
                index=iter_index,
            )
        plt.close("all")

    def _callback_firstiter_plot(
        self,
        output: "DataTypes",
        inputs: tuple["DataTypes", ...] | None = None,
        show: bool = False,
        save_path: str | pathlib.Path | None = None,
    ) -> None:
        if self._done_flag_firstiter_plot:
            return

        plotters_for_generator_out = get_methods_from_feature_classes_by_startswith_str(
            self, "firstiter_plot_"
        )

        for plotter in plotters_for_generator_out:
            execute_callback(
                self.logger,
                plotter,
                output,
                inputs,
                show=show,
                save_path=save_path,
            )
        plt.close("all")
