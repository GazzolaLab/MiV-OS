from __future__ import annotations

__doc__ = """
Here, we define the behavior of basic operator class, and useful mixin classes that can
be used to create new operators that conform to required behaviors.
"""
__all__ = [
    "GeneratorOperatorMixin",
]

from typing import TYPE_CHECKING, Callable, Generator, List, Optional, Protocol, Union

import functools
import inspect
import itertools
import os
import pathlib
from dataclasses import dataclass

if TYPE_CHECKING:
    from miv.core.datatype import DataTypes

from miv.core.operator.cachable import DataclassCacher
from miv.core.operator.operator import OperatorMixin
from miv.core.policy import VanillaRunner, _Runnable, _RunnerProtocol


class GeneratorOperatorMixin(OperatorMixin):
    def __init__(self):
        super().__init__()
        self.runner = VanillaRunner()
        self.cacher = DataclassCacher(self)

        assert self.tag != ""
        self.set_save_path("results")  # Default analysis path

    def output(self, skip_plot=False):
        """
        Output viewer. If cache exist, read result from cache value.
        Otherwise, execute (__call__) the module and return the value.
        """
        if self.cacher.check_cached():
            self.cacher.cache_called = True
            self.logger.info(f"Using cache: {self.cacher.cache_dir}")

            def generator_func():
                yield from self.cacher.load_cached()

            output = generator_func()
        else:
            self._cache_called = False
            self.logger.info("Cache not found.")
            self.logger.info(f"Using runner: {self.runner.__class__} type.")
            args = self.receive(skip_plot=skip_plot)  # Receive data from upstream
            if len(args) == 0:
                output = self.runner(self.__call__)
            else:
                output = self.runner(self.__call__, args)

        # Callback: After-run
        output = self.callback_after_run(output)
        return output
