__doc__ = """"""
__all__ = ["Operator", "DataLoader"]

from typing import Callable, Generator, List, Optional, Protocol, Union

from dataclasses import dataclass

from miv.core.operator.cachable import _Cachable
from miv.core.operator.callback import _Callback
from miv.core.operator.chainable import BaseChainingMixin, _Chainable
from miv.typing import DataTypes


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
    def load(self) -> Generator[DataTypes]:
        ...


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
        self._flag_executed = False

    def receive(self) -> List[DataTypes]:
        self._dependency_clear()
        return [node.output() for node in self.iterate_upstream()]

    @property
    def output(self) -> List[DataTypes]:
        if self._output is None:
            raise RuntimeError(f"{self} is not yet executed.")
        return self._output

    @property
    def finished(self) -> bool:
        return self._flag_executed

    def run(self) -> None:
        args = self.receive()
        self._output = self.run_policy.execute(*args)
        self._flag_executed = True

    def callback_before_run(self):
        pass

    def callback_after_run(self):
        pass

    def _assert_dependency_clear(self):
        """Assert upstream dependencies are finished"""
        for node in self.iterate_upstream():
            assert node.finished
