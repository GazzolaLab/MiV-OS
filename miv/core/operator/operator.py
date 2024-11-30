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
    "DataNodeMixin",
    "DataNode",
]

from typing import TYPE_CHECKING, List, Optional, Protocol, Union
from collections.abc import Callable, Generator

import functools
import inspect
import itertools
import os
import pathlib
from dataclasses import dataclass

if TYPE_CHECKING:
    from miv.core.datatype import DataTypes

from miv.core.operator.cachable import (
    CACHE_POLICY,
    DataclassCacher,
    FunctionalCacher,
    _Cachable,
    _CacherProtocol,
)
from miv.core.operator.callback import BaseCallbackMixin, _Callback
from miv.core.operator.chainable import BaseChainingMixin, _Chainable
from miv.core.operator.loggable import DefaultLoggerMixin, _Loggable
from miv.core.operator.policy import VanillaRunner, _Runnable, _RunnerProtocol


class Operator(
    _Callback,
    _Chainable,
    _Cachable,
    _Runnable,
    _Loggable,
    Protocol,
):
    """ """

    def run(self) -> None: ...


class DataLoader(
    _Callback,
    _Chainable,
    _Cachable,
    _Runnable,
    _Loggable,
    Protocol,
):
    """ """

    def load(self) -> Generator[DataTypes]: ...


class DataNode(_Chainable, _Runnable, _Loggable, Protocol): ...


class DataNodeMixin(BaseChainingMixin, DefaultLoggerMixin):
    """ """

    def __init__(self):
        super().__init__()

    def output(self) -> list[DataTypes]:
        return self

    def run(self, *args, **kwargs):
        return self.output()


class DataLoaderMixin(BaseChainingMixin, BaseCallbackMixin, DefaultLoggerMixin):
    """ """

    def __init__(self):
        super().__init__()

        self.runner = VanillaRunner()
        self.cacher = FunctionalCacher(self)

        self.tag = "data_loader"
        self.set_save_path(self.data_path)  # Default analysis path

        self._load_param = {}
        self.skip_plot = True

    def configure_load(self, **kwargs):
        """
        (Experimental Feature)
        """
        self._load_param = kwargs

    def output(self) -> list[DataTypes]:
        output = self.load(**self._load_param)
        if not self.skip_plot:
            # if output is generator, raise error
            if inspect.isgenerator(output):
                raise ValueError(
                    "output() method of DataLoaderMixin cannot support generator type."
                )
            self.make_analysis_path()
            self.plot(output, None, show=False, save_path=True)
        return output

    def run(self, *args, **kwargs):
        return self.output()

    def load(self):
        raise NotImplementedError("load() method must be implemented to be DataLoader.")


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

    def __init__(self) -> None:
        super().__init__()
        self.runner = VanillaRunner()
        self.cacher = DataclassCacher(self)

        assert self.tag != ""
        self.set_save_path("results")  # Default analysis path

    def __repr__(self):
        return self.tag

    def __str__(self):
        return self.tag

    def set_caching_policy(self, policy: CACHE_POLICY) -> None:
        self.cacher.policy = policy

    def receive(self) -> list[DataTypes]:
        """
        Receive input data from each upstream operator.
        Essentially, this method recursively call upstream operators' run() method.
        """
        return [node.run(skip_plot=self.skip_plot) for node in self.iterate_upstream()]

    def output(self):
        """
        Output viewer. If cache exist, read result from cache value.
        Otherwise, execute (__call__) the module and return the value.
        """
        cache_called = self.cacher.check_cached()
        if cache_called:
            self.logger.info(f"Using cache: {self.cacher.cache_dir}")
            loader = self.cacher.load_cached()
            output = next(loader, None)
        else:
            self.logger.info("Cache not found.")
            self.logger.info(f"Using runner: {self.runner.__class__} type.")
            args = self.receive()  # Receive data from upstream
            if len(args) == 0:
                output = self.runner(self.__call__)
            else:
                output = self.runner(self.__call__, args)

            # Callback: After-run
            self.callback_after_run(output)

            # Plotting: Only happened when cache is not called
            if not self.skip_plot:
                if len(args) == 0:
                    self.plot(output, None, show=False, save_path=True)
                elif len(args) == 1:
                    self.plot(output, args[0], show=False, save_path=True)
                else:
                    self.plot(output, args, show=False, save_path=True)

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
        self.skip_plot = skip_plot

        output = self.output()

        return output
