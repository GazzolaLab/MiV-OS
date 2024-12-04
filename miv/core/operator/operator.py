from __future__ import annotations

__doc__ = """
Here, we define the behavior of basic operator class, and useful mixin classes that can
be used to create new operators that conform to required behaviors.
"""
__all__ = [
    "DataLoader",
    "DataLoaderMixin",
    "OperatorMixin",
    "DataNodeMixin",
    "LoaderNode",
]

from typing import TYPE_CHECKING, List, Optional, Protocol, Union, Any, cast
from collections.abc import Iterator
from collections.abc import Callable, Generator
from typing_extensions import Self

import functools
import inspect
import itertools
import os
import logging
import pathlib
from dataclasses import dataclass

if TYPE_CHECKING:
    from miv.core.datatype import DataTypes
    from miv.core.datatype.signal import Signal
    from miv.core.datatype.spikestamps import Spikestamps

from miv.core.operator.cachable import (
    DataclassCacher,
    FunctionalCacher,
    _CacherProtocol,
    CACHE_POLICY,
)
from miv.core.operator.callback import BaseCallbackMixin
from miv.core.operator.chainable import BaseChainingMixin
from miv.core.operator.loggable import DefaultLoggerMixin
from miv.core.operator.policy import VanillaRunner, _RunnerProtocol

from ..protocol import _Loggable
from .protocol import _Runnable, _Chainable, _Cachable, _Callback, OperatorNode


class DataLoader(
    _Callback,
    _Chainable,
    Protocol,
):
    """ """

    def load(
        self, *args: Any, **kwargs: Any
    ) -> Generator[DataTypes] | Spikestamps | Generator[Signal]: ...


class LoaderNode(_Chainable, _Runnable, Protocol): ...


class DataNodeMixin(BaseChainingMixin, DefaultLoggerMixin):
    """ """

    data: DataTypes

    def output(self) -> Self:
        return self

    def run(self) -> Self:
        return self.output()


class DataLoaderMixin(BaseChainingMixin, DefaultLoggerMixin):
    """ """

    def __init__(self) -> None:
        self.tag = "data_loader"
        self._cacher: _CacherProtocol = FunctionalCacher(self)
        self.runner = VanillaRunner()
        super().__init__()

        self._load_param: dict = {}

    def __call__(self) -> DataTypes:
        raise NotImplementedError("Please implement __call__ method.")

    @property
    def cacher(self) -> _CacherProtocol:
        return self._cacher

    @cacher.setter
    def cacher(self, value: _CacherProtocol) -> None:
        # FIXME:
        policy = self._cacher.policy
        cache_dir = self._cacher.cache_dir
        self._cacher = value
        self._cacher.policy = policy
        self._cacher.cache_dir = cache_dir

    def set_caching_policy(self, policy: CACHE_POLICY) -> None:
        self.cacher.policy = policy

    def configure_load(self, **kwargs: Any) -> None:
        """
        (Experimental Feature)
        """
        self._load_param = kwargs

    def output(self) -> Generator[DataTypes] | Spikestamps | Generator[Signal]:
        output = self.load(**self._load_param)
        return output

    def run(self) -> DataTypes:
        return self.output()

    def load(
        self, *args: Any, **kwargs: Any
    ) -> Generator[DataTypes] | Spikestamps | Generator[Signal]:
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
        self.runner: _RunnerProtocol = VanillaRunner()
        self._cacher: _CacherProtocol = DataclassCacher(self)

        self.analysis_path = "analysis"
        self.tag = "operator"

        super().__init__()

    def __call__(self) -> DataTypes:
        raise NotImplementedError("Please implement __call__ method.")

    @property
    def cacher(self) -> _CacherProtocol:
        return self._cacher

    @cacher.setter
    def cacher(self, value: _CacherProtocol) -> None:
        # FIXME:
        policy = self._cacher.policy
        cache_dir = self._cacher.cache_dir
        self._cacher = value
        self._cacher.policy = policy
        self._cacher.cache_dir = cache_dir

    def set_caching_policy(self, policy: CACHE_POLICY) -> None:
        self.cacher.policy = policy

    def __repr__(self) -> str:
        return self.tag

    def __str__(self) -> str:
        return self.tag

    def receive(self) -> list[DataTypes]:
        """
        Receive input data from each upstream operator.
        Essentially, this method recursively call upstream operators' run() method.
        """
        return [cast(OperatorNode, node).run() for node in self.iterate_upstream()]

    def output(self) -> DataTypes:
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

    def run(self) -> DataTypes:
        """
        Execute the module. This is the function called by the pipeline.
        Input to the parameters are received from upstream operators.
        """
        output = self.output()

        return output
