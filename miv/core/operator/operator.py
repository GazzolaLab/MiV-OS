from __future__ import annotations

__doc__ = """
Here, we define the behavior of basic operator class, and useful mixin classes that can
be used to create new operators that conform to required behaviors.
"""
__all__ = [
    "Operator",
    "DataLoader",
    "DataLoaderMixin",
    "OperatorMixin",
    "GeneratorOperatorMixin",
    "DataNodeMixin",
    "DataNode",
]

from typing import TYPE_CHECKING, Callable, Generator, List, Optional, Protocol, Union

import functools
import inspect
import itertools
import os
import pathlib
from dataclasses import dataclass

if TYPE_CHECKING:
    from miv.core.datatype import DataTypes

from miv.core.operator.cachable import (
    DataclassCacher,
    FunctionalCacher,
    _Cachable,
    _CacherProtocol,
)
from miv.core.operator.callback import BaseCallbackMixin, _Callback
from miv.core.operator.chainable import BaseChainingMixin, _Chainable
from miv.core.operator.loggable import DefaultLoggerMixin, _Loggable
from miv.core.policy import VanillaRunner, _Runnable, _RunnerProtocol


class Operator(
    _Callback,
    _Chainable,
    _Cachable,
    _Runnable,
    _Loggable,
    Protocol,
):
    """ """

    def run(self) -> None:
        ...

    def set_save_path(self, path: str | pathlib.Path, recursive: bool = False) -> None:
        ...


class DataLoader(
    _Callback,
    _Chainable,
    _Cachable,
    _Runnable,
    _Loggable,
    Protocol,
):
    """ """

    def load(self) -> Generator[DataTypes, None, None]:
        ...


class DataNode(_Chainable, _Runnable, _Loggable, Protocol):
    ...


class DataNodeMixin(BaseChainingMixin, DefaultLoggerMixin):
    """ """

    def __init__(self):
        super().__init__()
        self._output = None

    @property
    def output(self) -> list[DataTypes]:
        return self._output

    def run(self, *args, **kwargs):
        self._output = self
        return self._output

    def set_save_path(self, path: str | pathlib.Path, recursive: bool = False) -> None:
        pass


class DataLoaderMixin(BaseChainingMixin, BaseCallbackMixin, DefaultLoggerMixin):
    """ """

    def __init__(self):
        super().__init__()
        self._output: DataTypes | None = None

        self.runner = VanillaRunner()
        self.cacher = FunctionalCacher(self)
        self.cacher.cache_dir = os.path.join(self.data_path, ".cache")

        self._load_param = {}

    def configure_load(self, **kwargs):
        """
        (Experimental Feature)
        """
        self._load_param = kwargs

    @property
    def output(self) -> list[DataTypes]:
        return self._output

    def run(self, *args, **kwargs):
        self._output = self.load(**self._load_param)
        return self._output

    def set_save_path(self, path: str | pathlib.Path, recursive: bool = False) -> None:
        pass


class OperatorMixin(BaseChainingMixin, BaseCallbackMixin, DefaultLoggerMixin):
    """
    Behavior includes:
        - Whenever "run()" method is executed:
            1. Check if the module is cached in the same parameters. Y: Pass to 6
            2. Callback: before run
            3. Run
            4. Callback: after run
            5. Save cache
            6. Exit
        - Cache includes:
            - All results from callback
    """

    def __init__(self):
        super().__init__()
        self.runner = VanillaRunner()
        self.cacher = DataclassCacher(self)

        assert self.tag != ""
        self.set_save_path("results")  # Default analysis path

    def set_caching_policy(self, cacher: _CacherProtocol):
        self.cacher = cacher(self)

    def set_save_path(self, path: str | pathlib.Path, recursive: bool = False):
        self.analysis_path = os.path.join(path, self.tag.replace(" ", "_"))
        self.cacher.cache_dir = os.path.join(self.analysis_path, ".cache")
        if recursive:
            # TODO: if circular dependency exists, this will cause infinite loop
            for node in self.iterate_upstream():
                node.set_save_path(path, recursive=True)

    def make_analysis_path(self):
        os.makedirs(self.analysis_path, exist_ok=True)

    def receive(self, skip_plot=False) -> list[DataTypes]:
        """
        Receive input data from each upstream operator.
        Essentially, this method recursively call upstream operators' run() method.
        """
        return [node.run(skip_plot=skip_plot) for node in self.iterate_upstream()]

    def output(self, skip_plot=False):
        """
        Output viewer. If cache exist, read result from cache value.
        Otherwise, execute (__call__) the module and return the value.
        """
        if self.cacher.check_cached():
            self.cacher.cache_called = True
            loader = self.cacher.load_cached()
            output = next(loader, None)
        else:
            self.cacher.cache_called = False
            args = self.receive(skip_plot=skip_plot)  # Receive data from upstream
            if len(args) == 0:
                output = self.runner(self.__call__)
            else:
                output = self.runner(self.__call__, args)

        # Callback: After-run
        output = self.callback_after_run(output)
        return output

    def run(
        self,
        skip_plot: bool = False,
    ) -> None:
        """
        Execute the module. This is the function called by the pipeline.
        Input to the parameters are received from upstream operators.
        """
        self.make_analysis_path()
        output = self.output(skip_plot=skip_plot)

        # Plotting
        if not skip_plot and not self.cacher.cache_called:
            self.plot(show=False, save_path=True)

        return output


class GeneratorOperatorMixin(OperatorMixin):
    def output(self, skip_plot=False):
        """
        Output viewer. If cache exist, read result from cache value.
        Otherwise, execute (__call__) the module and return the value.
        """
        if self.cacher.check_cached():
            self.cacher.cache_called = True
            self.logger.info(f"Using cache: {self.cacher.cache_dir}")

            def generator_func():
                yield from self.cacher.load_cached()

            output = generator_func()
        else:
            self._cache_called = False
            self.logger.info("Cache not found.")
            self.logger.info(f"Using runner: {self.runner.__class__} type.")
            args = self.receive(skip_plot=skip_plot)  # Receive data from upstream
            if len(args) == 0:
                output = self.runner(self.__call__)
            else:
                output = self.runner(self.__call__, args)

        # Callback: After-run
        output = self.callback_after_run(output)
        return output
