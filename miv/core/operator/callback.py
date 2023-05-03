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
        self.skip_plotting: bool = False

    def __lshift__(self, right: Callable) -> SelfCallback:
        if right.__name__.startswith(
            "__prepend"
        ):  # TODO: need better way to prepend callbacks
            self._callback_before_run.append(right)
            return self
        if right.__name__.startswith("plot_"):
            self._callback_plot.append(right)
            return self
        self._callback_after_run.append(right)
        return self

    def callback_before_run(self, inputs):
        predefined_callbacks = get_methods_from_feature_classes_by_startswith_str(
            self, "before_run"
        )
        for callback in predefined_callbacks + self._callback_before_run:
            inputs = callback(self, inputs)
        return inputs

    def callback_after_run(self, output):
        predefined_callbacks = get_methods_from_feature_classes_by_startswith_str(
            self, "after_run"
        )
        for callback in predefined_callbacks + self._callback_after_run:
            output = callback(self, output)
        return output

    def plot_from_callbacks(self, *args, **kwargs):
        for callback in self._callback_plot:
            callback(self, *args, **kwargs)

    def plot(
        self,
        show: bool = False,
        save_path: Optional[Union[bool, str, pathlib.Path]] = None,
        dry_run: bool = False,
    ):
        if self.skip_plotting:
            return
        if save_path is True:
            os.makedirs(self.analysis_path, exist_ok=True)
            save_path = self.analysis_path
        plotters = get_methods_from_feature_classes_by_startswith_str(self, "plot")
        if dry_run:
            for plotter in plotters:
                print(f"dry run: {plotter}")
            return
        plotters_for_generator_out = get_methods_from_feature_classes_by_startswith_str(
            self, "_generator_plot_"
        )
        if len(plotters_for_generator_out) > 0:  # TODO: Experimental work
            if inspect.isgenerator(self._output):
                inputs = self.receive()
                for index, (output_seg, zipped_inputs) in enumerate(
                    zip(self._output, zip(*inputs))
                ):
                    for plotter in plotters_for_generator_out:
                        plotter(
                            self,
                            output_seg,
                            show=show,
                            save_path=save_path,
                            index=index,
                            zipped_inputs=zipped_inputs,
                        )
            else:
                zipped_inputs = self.receive()
                for plotter in plotters_for_generator_out:
                    plotter(
                        self,
                        self._output,
                        show=show,
                        save_path=save_path,
                        index=0,
                        zipped_inputs=zipped_inputs,
                    )
        for plotter in plotters:
            plotter(self, self._output, show=show, save_path=save_path)
        if not show:
            plt.close("all")
