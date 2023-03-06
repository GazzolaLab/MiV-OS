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
        assert self.tag != ""
        self.analysis_path = os.path.join(
            "results", self.tag.replace(" ", "_")
        )  # Default analysis path

        self.runner = VanillaRunner()
        self.cacher = DataclassCacher(self)

    def set_caching_policy(self, cacher: _CacherProtocol):
        self.cacher = cacher(self)

    def receive(self) -> list[DataTypes]:
        return [node.output for node in self.iterate_upstream()]

    @property
    def output(self) -> list[DataTypes]:
        # FIXME: Run graph upstream? what about the order??
        if self._output is None:
            raise RuntimeError(f"{self} is not yet executed.")
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
        save_path: str | pathlib.Path,
        dry_run: bool = False,
        cache_dir: str | pathlib.Path = ".cache",
        enable_save_plot: bool = True,
    ) -> None:
        # Execute the module
        self.analysis_path = os.path.join(save_path, self.tag.replace(" ", "_"))
        self.cacher.cache_dir = os.path.join(self.analysis_path, cache_dir)

        if dry_run:
            print("Dry run: ", self.__class__.__name__)
            return
        self._execute()
        self.plot(save_path=True, dry_run=dry_run)
