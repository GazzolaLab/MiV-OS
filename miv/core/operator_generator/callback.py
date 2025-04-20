__doc__ = """"""

from typing import TYPE_CHECKING, Any, Protocol

import pathlib

import matplotlib.pyplot as plt

from miv.core.operator.callback import (
    get_methods_from_feature_classes_by_startswith_str,
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


class GeneratorCallbackMixin:
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
        super().__init__(*args, **kwargs)
        self._done_flag_generator_plot = getattr(kwargs, "plot", False)
        self._done_flag_firstiter_plot = getattr(kwargs, "plot", False)

    def _callback_generator_plot(
        self,
        iter_index: int,
        output: "DataTypes",
        inputs: tuple["DataTypes", ...] | None = None,
        show: bool = False,
        save_path: bool | str | pathlib.Path | None = None,
    ) -> None:
        if self._done_flag_generator_plot:
            return

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
            plotter(
                output,
                inputs,
                show=show,
                save_path=save_path,
            )
        plt.close("all")
