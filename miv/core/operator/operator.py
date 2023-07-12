from __future__ import annotations

__doc__ = """"""
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
from miv.core.policy import VanillaRunner, _Runnable, _RunnerProtocol


class Operator(
    _Callback,
    _Chainable,
    _Cachable,
    _Runnable,
    Protocol,
):
    """ """

    def run(self, dry_run: bool = False) -> None:
        ...

    def set_save_path(self, path: str | pathlib.Path, recursive: bool = False) -> None:
        ...


class DataLoader(
    _Callback,
    _Chainable,
    _Cachable,
    _Runnable,
    Protocol,
):
    """ """

    def load(self) -> Generator[DataTypes, None, None]:
        ...


class DataNode(_Chainable, _Runnable, Protocol):
    ...


class DataNodeMixin(BaseChainingMixin):
    """ """

    def __init__(self):
        super().__init__()
        self._output = None

    @property
    def output(self) -> list[DataTypes]:
        self._output = self
        return self._output

    def run(self, **kwargs):
        pass

    def set_save_path(self, path: str | pathlib.Path, recursive: bool = False) -> None:
        pass


class DataLoaderMixin(BaseChainingMixin, BaseCallbackMixin):
    """ """

    def __init__(self):
        super().__init__()
        self._output: DataTypes | None = None

        self.runner = VanillaRunner()
        self.cacher = FunctionalCacher(self)
        self.cacher.cache_dir = os.path.join(self.data_path, ".cache")

    @property
    def output(self) -> list[DataTypes]:
        self._output = self.load()
        return self._output

    def run(self, **kwargs):
        pass

    def set_save_path(self, path: str | pathlib.Path, recursive: bool = False) -> None:
        pass


class OperatorMixin(BaseChainingMixin, BaseCallbackMixin):
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
        self._output: DataTypes | None = None
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

    def receive(self) -> list[DataTypes]:
        return [node.output for node in self.iterate_upstream()]

    @property
    def output(self) -> list[DataTypes]:
        self._execute()
        return self._output  # TODO: Just use upstream caller instead of .output

    def _execute(self):
        args: list[DataTypes] = self.receive()  # Receive data from upstream
        # TODO : implement pre-run callback
        # args = self.callback_before_run(args)
        if len(args) == 0:
            output = self.runner(self.__call__)
        else:
            output = self.runner(self.__call__, args)
        # Post Processing
        self._output = self.callback_after_run(output)

    def run(
        self,
        dry_run: bool = False,
        skip_plot: bool = False,
    ) -> None:
        # Execute the module
        if dry_run:
            print("Dry run: ", self.__class__.__name__)
            return
        self.make_analysis_path()
        self._execute()
        cache_called = self.cacher.cache_called
        # TODO
        if not skip_plot and not cache_called and not self.pipeline_called:
            self.plot(show=False, save_path=True, dry_run=dry_run)
