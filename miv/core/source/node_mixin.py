from __future__ import annotations

__doc__ = """
Mixin to create a data loader node.
"""

from typing import TYPE_CHECKING, Any
from collections.abc import Generator

from ..chainable import ChainingMixin
from ..operator.callback import BaseCallbackMixin
from ..loggable import DefaultLoggerMixin
from ..operator.policy import VanillaRunner
from .cachable import FunctionalCacher

if TYPE_CHECKING:
    from ..datatype import DataTypes
    from ..datatype.signal import Signal
    from ..datatype.spikestamps import Spikestamps


class DataLoaderMixin(ChainingMixin, BaseCallbackMixin, DefaultLoggerMixin):
    """ """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.runner = VanillaRunner()
        super().__init__(*args, cacher=FunctionalCacher(self), **kwargs)
        self._load_param: dict = {}

        # Attribute from upstream
        self.tag: str

    def __call__(self) -> DataTypes:
        raise NotImplementedError("Please implement __call__ method.")

    def configure_load(self, **kwargs: Any) -> None:
        """
        (Experimental Feature)
        """
        self._load_param = kwargs

    def flow_blocked(self) -> bool:
        return False

    def output(self) -> Generator[DataTypes] | Spikestamps | Generator[Signal]:
        output = self.load(**self._load_param)
        return output

    def load(
        self, *args: Any, **kwargs: Any
    ) -> Generator[DataTypes] | Spikestamps | Generator[Signal]:
        raise NotImplementedError("load() method must be implemented to be DataLoader.")
