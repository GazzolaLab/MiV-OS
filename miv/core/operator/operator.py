__doc__ = """"""
__all__ = ["Operator", "DataLoader", "DataLoaderMixin", "OperatorMixin"]

from typing import Callable, Generator, List, Optional, Protocol, Union

import pathlib
from dataclasses import dataclass

from miv.core.datatype import DataTypes
from miv.core.operator.cachable import _Cachable
from miv.core.operator.callback import _Callback
from miv.core.operator.chainable import BaseChainingMixin, _Chainable


class Operator(
    _Callback,
    _Chainable,
    _Cachable,
    # Callable[[DataTypes],DataTypes],
    Protocol,
):
    @dataclass
    class Config:
        pass

    def get_config(self) -> Config:
        ...

    def run(self, *args) -> DataTypes:
        ...

    def query(self) -> List[DataTypes]:
        ...


class DataLoader(
    _Callback,
    _Chainable,
    Protocol,
):
    def load(self) -> Generator[DataTypes, None, None]:
        ...


class DataLoaderMixin(BaseChainingMixin):
    """ """

    def __init__(self):
        super().__init__()
        self._output: Optional[DataTypes] = None

    @property
    def output(self) -> List[DataTypes]:
        if self._output is None:
            raise RuntimeError(f"{self} is not yet executed.")
        return self._output

    def run(self, analysis_path: Optional[Union[str, pathlib.Path]] = None) -> None:
        if analysis_path is None and not hasattr(self, "analysis_path"):
            raise ValueError(f"Please specify {analysis_path=}.")
        self._output = self.load()

    def callback_before_run(self):
        pass

    def callback_after_run(self):
        pass


class OperatorMixin(BaseChainingMixin):
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

    def receive(self) -> List[DataTypes]:
        self._dependency_clear()
        return [node.output() for node in self.iterate_upstream()]

    @property
    def output(self) -> List[DataTypes]:
        if self._output is None:
            raise RuntimeError(f"{self} is not yet executed.")
        return self._output

    def run(self) -> None:
        # Execute the module
        args = self.receive()
        if len(args) == 0:
            self._output = self.__call__()
        else:
            self._output = self.__call__(*args)

    def callback_before_run(self):
        pass

    def callback_after_run(self):
        pass
