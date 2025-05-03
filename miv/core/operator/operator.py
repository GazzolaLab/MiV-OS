from __future__ import annotations

__doc__ = """
Here, we define the behavior of basic operator class, and useful mixin classes that can
be used to create new operators that conform to required behaviors.
"""
__all__ = [
    "DataLoaderMixin",
    "DataNodeMixin",
    "OperatorMixin",
]

from typing import TYPE_CHECKING, Any, cast
from collections.abc import Generator
from typing_extensions import Self

import pathlib


from miv.core.operator.cachable import (
    DataclassCacher,
    FunctionalCacher,
)
from miv.core.operator.callback import BaseCallbackMixin
from miv.core.operator.chainable import BaseChainingMixin
from miv.core.operator.loggable import DefaultLoggerMixin
from miv.core.operator.policy import VanillaRunner, _RunnerProtocol

if TYPE_CHECKING:
    from miv.core.datatype import DataTypes
    from miv.core.datatype.signal import Signal
    from miv.core.datatype.spikestamps import Spikestamps
    from .protocol import _Node

else:

    class _Node: ...


class DataNodeMixin(BaseChainingMixin, DefaultLoggerMixin):
    """ """

    data: DataTypes

    def output(self) -> Self:
        return self

    def run(self) -> Self:
        return self.output()


class DataLoaderMixin(BaseChainingMixin, BaseCallbackMixin, DefaultLoggerMixin):
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

    def output(self) -> Generator[DataTypes] | Spikestamps | Generator[Signal]:
        output = self.load(**self._load_param)
        return output

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.runner: _RunnerProtocol = VanillaRunner()
        self.analysis_path = "analysis"

        super().__init__(*args, cacher=DataclassCacher(self), **kwargs)

        # Attribute from upstream
        self.tag: str

    def __call__(self) -> DataTypes:
        raise NotImplementedError("Please implement __call__ method.")

    def __repr__(self) -> str:
        return self.tag

    def __str__(self) -> str:
        return self.tag

    def receive(self) -> list[DataTypes]:
        """
        Receive input data from each upstream operator.
        Essentially, this method recursively call upstream operators' run() method.
        """
        return [cast(_Node, node).output() for node in self.iterate_upstream()]

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
