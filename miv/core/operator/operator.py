from __future__ import annotations

__doc__ = """
Here, we define the behavior of basic operator class, and useful mixin classes that can
be used to create new operators that conform to required behaviors.
"""

from typing import TYPE_CHECKING, Any, cast
from abc import abstractmethod

import pathlib

from loguru import logger
from ..chainable import ChainingMixin
from ..loggable import DefaultLoggerMixin
from .cachable import DataclassCacher
from .callback import BaseCallbackMixin
from .policy import VanillaRunner, RunnerBase

if TYPE_CHECKING:
    from ..datatype import DataTypes
    from .protocol import _Node

else:

    class _Node: ...


class OperatorMixin(ChainingMixin, BaseCallbackMixin, DefaultLoggerMixin):
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
        self.runner: RunnerBase = VanillaRunner()
        self.analysis_path = "analysis"

        super().__init__(*args, cacher=DataclassCacher(self), **kwargs)

        # Attribute from upstream
        self.tag: str

    @abstractmethod
    def __call__(self) -> DataTypes:
        """Execute the operator. Must be implemented by subclasses."""

    def __repr__(self) -> str:
        return self.tag

    def __str__(self) -> str:
        return self.tag

    def receive(self) -> list[DataTypes]:
        """
        Receive input data from each upstream operator.
        Essentially, this method recursively call upstream operators' run() method.
        """
        ret = []
        for node in self.iterate_upstream():
            output = cast(_Node, node).output()
            ret.append(output)
        return ret

    def flow_blocked(self) -> bool:
        try:
            return self.cacher.check_cached(skip_log=True)
        except (AttributeError, FileNotFoundError):
            return False

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

            # Save cache after execution (respects policy via when_policy_is decorator)
            if output is not None:
                self.cacher.save_cache(output, tag="data")
                self.cacher.save_config(tag="data")

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
