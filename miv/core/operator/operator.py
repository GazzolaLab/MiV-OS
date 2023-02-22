__doc__ = """"""
__all__ = ["Operator", "DataLoader", "DataLoaderMixin", "OperatorMixin"]

from typing import Callable, Generator, List, Optional, Protocol, Union

import functools
import itertools
import os
import pathlib
from dataclasses import dataclass

from miv.core.datatype import DataTypes
from miv.core.operator.cachable import DataclassCacher, _Cachable, _CacherProtocol
from miv.core.operator.callback import BaseCallbackMixin, _Callback
from miv.core.operator.chainable import BaseChainingMixin, _Chainable
from miv.core.policy import VanillaRunner, _Runnable, _RunnerProtocol


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


class Operator(
    _Callback,
    _Chainable,
    _Cachable,
    _Runnable,
    # Callable[[DataTypes],DataTypes],
    Protocol,
):
    @dataclass
    class Config:
        pass

    def get_config(self) -> Config:
        ...

    def run(self, dry_run: bool = False) -> None:
        ...

    def query(self) -> List[DataTypes]:
        ...


class DataLoader(
    _Callback,
    _Chainable,
    _Runnable,
    Protocol,
):
    """ """

    def load(self) -> Generator[DataTypes, None, None]:
        ...

    def run(self, dry_run: bool = False) -> None:
        ...


class DataLoaderMixin(BaseChainingMixin, BaseCallbackMixin):
    """ """

    def __init__(self):
        super().__init__()
        self._output: Optional[DataTypes] = None

        self.runner = VanillaRunner()

    @property
    def output(self) -> List[DataTypes]:
        self._output = self.load()
        return self._output

    def run(self, dry_run: bool = False) -> None:
        if dry_run:
            print("Dry run: ", self.__class__.__name__)
            return


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
        self._output: Optional[DataTypes] = None
        assert self.tag != ""
        self.analysis_path = os.path.join(
            "results", self.tag.replace(" ", "_")
        )  # Default analysis path

        self.runner = VanillaRunner()
        self.cacher = DataclassCacher(self)

    def set_caching_policy(self, cacher: _CacherProtocol):
        self.cacher = cacher(self)

    def receive(self) -> List[DataTypes]:
        return [node.output for node in self.iterate_upstream()]

    @property
    def output(self) -> List[DataTypes]:
        # FIXME: Run graph upstream? what about the order??
        if self._output is None:
            raise RuntimeError(f"{self} is not yet executed.")
        self._execute()
        return self._output  # TODO: Just use upstream caller instead of .output

    def _execute(self):
        args: List[DataTypes] = self.receive()  # Receive data from upstream
        if len(args) == 0:
            self._output = self.runner(self.__call__)
        else:
            self._output = self.runner(self.__call__, args)

    def run(
        self,
        save_path: Union[str, pathlib.Path] = "results",
        dry_run: bool = False,
        cache_dir: Union[str, pathlib.Path] = ".cache",
        enable_callback: bool = True,
    ) -> None:
        # Execute the module
        self.analysis_path = os.path.join(save_path, self.tag.replace(" ", "_"))
        self.cacher.cache_dir = os.path.join(self.analysis_path, cache_dir)

        if enable_callback:
            self.callback_before_run()

        if dry_run:
            print("Dry run: ", self.__class__.__name__)
            return
        self._execute()

        # Post Processing
        if enable_callback:
            self.callback_after_run()

    def plot(
        self,
        show: bool = False,
        save_path: Optional[Union[bool, str, pathlib.Path]] = None,
        dry_run: bool = False,
    ):
        if save_path is True:
            os.makedirs(self.analysis_path, exist_ok=True)
            save_path = self.analysis_path
        plotters = get_methods_from_feature_classes_by_startswith_str(self, "plot")
        if dry_run:
            for plotter in plotters:
                print(f"dry run: {plotter}")
            return
        for plotter in plotters:
            plotter(self, self._output, show=show, save_path=save_path)
