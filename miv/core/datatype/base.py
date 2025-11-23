from __future__ import annotations

__doc__ = """
Base mixin for datatype classes that provides chaining capabilities and makes them
compatible with the pipeline system.
"""
__all__ = ["DataNodeMixin"]

from typing import TYPE_CHECKING, Any
from typing_extensions import Self

from ..chainable import ChainingMixin
from ..operator.loggable import DefaultLoggerMixin

if TYPE_CHECKING:
    from . import DataTypes
else:
    DataTypes = Any


class DataNodeMixin(ChainingMixin, DefaultLoggerMixin):
    """Base mixin for datatype classes.

    Provides chaining capabilities and makes datatypes compatible with the pipeline system.
    All datatypes (Signal, Spikestamps, Events, etc.) inherit from this mixin.
    """

    data: DataTypes

    def flow_blocked(self) -> bool:
        return False

    def output(self) -> Self:
        return self

    def run(self) -> Self:
        return self.output()
