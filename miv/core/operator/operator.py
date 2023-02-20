__doc__ = """"""
__all__ = ["Operator", "DataLoader", "DataLoaderMixin", "OperatorMixin"]

from typing import Callable, Generator, List, Optional, Protocol, Union

import functools
import pathlib
from dataclasses import dataclass

from miv.core.datatype import DataTypes
from miv.core.operator.cachable import DataclassCachableMixin, _Cachable
from miv.core.operator.callback import _Callback
from miv.core.operator.chainable import BaseChainingMixin, _Chainable
from miv.core.policy import VanillaRunner, _Runnable, _RunnerProtocol


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
    def load(self) -> Generator[DataTypes, None, None]:
        ...

    def run(self, dry_run: bool = False) -> None:
        ...


class DataLoaderMixin(BaseChainingMixin):
    """ """

    def __init__(self):
        super().__init__()
        self._output: Optional[DataTypes] = None

        self.runner = VanillaRunner()

    @property
    def output(self) -> List[DataTypes]:
        if self._output is None:
            raise RuntimeError(f"{self} is not yet executed.")
        return self._output

    def run(self, dry_run: bool = False) -> None:
        if dry_run:
            print("Dry run: ", self.__class__.__name__)
            return
        self._output = self.load()

    def callback_before_run(self):
        pass

    def callback_after_run(self):
        pass


class OperatorMixin(BaseChainingMixin, DataclassCachableMixin):
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

        self.runner = VanillaRunner()

    def receive(self) -> List[DataTypes]:
        return [node.output for node in self.iterate_upstream()]

    @property
    def output(self) -> List[DataTypes]:
        if self._output is None:
            raise RuntimeError(f"{self} is not yet executed.")
        return self._output

    def run(self, dry_run: bool = False) -> None:
        # Execute the module
        args: List[DataTypes] = self.receive()  # Receive data from upstream
        if dry_run:
            print("Dry run: ", self.__class__.__name__)
            return
        if len(args) == 0:
            self._output = self.runner(self.__call__)
        else:
            self._output = self.runner(self.__call__, args)

    def callback_before_run(self):
        pass

    def callback_after_run(self):
        pass
