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
from miv.core.operator.callback import BaseCallbackMixin
from miv.core.operator.chainable import BaseChainingMixin, _Chainable
from miv.core.operator.loggable import DefaultLoggerMixin, _Loggable
from miv.core.operator.policy import VanillaRunner, _Runnable, _RunnerProtocol


class _Callback(Protocol):
    def __lshift__(self, right: Callable) -> SelfCallaback: ...
    def set_save_path(
        self,
        path: Union[str, pathlib.Path],
        cache_path: Union[str, pathlib.Path] = None,
    ) -> None: ...

    def _reset_callbacks(
        self, *, after_run: bool = False, plot: bool = False
    ) -> None: ...
    def _callback_after_run(self) -> None: ...
    def _plot(self) -> None: ...


class Operator(
    _Callback,
    _Chainable,
    _Cachable,
    _Runnable,
    _Loggable,
    Protocol,
):
    """ """

    @property
    def tag(self) -> str: ...

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

    def load(self) -> Generator[DataTypes, None, None]: ...


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
        self.tag = "data_loader"
        self.cacher = FunctionalCacher(self)
        self.runner = VanillaRunner()
        super().__init__()


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

    def __init__(self):
        self.runner = VanillaRunner()
        self.cacher = DataclassCacher(self)

        super().__init__()

    def set_caching_policy(self, cacher: _CacherProtocol):
        self.cacher = cacher(self)

    def receive(self) -> list[DataTypes]:
        """
        Receive input data from each upstream operator.
        Essentially, this method recursively call upstream operators' run() method.
        """
        return [node.run() for node in self.iterate_upstream()]

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
            self._callback_after_run(output)

            # Plotting: Only happened when cache is not called
            self._callback_plot(output, args, show=False)

        return output

    def plot(
        self, show: bool = False, save_path: str | pathlib.Path | None = None
    ) -> None:
        """
        Standalone plotting operation.
        """
        cache_called = self.cacher.check_cached()
        if not cache_called:
            raise NotImplementedError(
                "Standalone plotting is only possible if this operator has"
                "results stored in cache. Please use Pipeline(op).run() first."
            )
        loader = self.cacher.load_cached()
        output = next(loader, None)

        # Plotting: Only happened when cache is not called
        args = self.receive()  # Receive data from upstream
        self._done_flag_plot = False  # FIXME
        self._callback_plot(output, args, show=show, save_path=save_path)

        return output

    def run(self) -> None:
        """
        Execute the module. This is the function called by the pipeline.
        Input to the parameters are received from upstream operators.
        """
        output = self.output()

        return output
