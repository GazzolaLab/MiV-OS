__doc__ = """"""
__all__ = ["Operator"]

from typing import Callable, List, Protocol, Union

from dataclasses import dataclass

from miv.core.operator.callback import _Callback
from miv.core.operator.chainable import _Chainable
from miv.typing import DataTypes


class Operator(
    _Callback,
    _Chainable,
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
