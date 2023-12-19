from __future__ import annotations

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
from miv.core.operator_generator.callback import GeneratorCallbackMixin
from miv.core.operator_generator.policy import VanillaGeneratorRunner


class GeneratorOperatorMixin(OperatorMixin, GeneratorCallbackMixin):
    def __init__(self):
        super().__init__()
        self.runner = VanillaGeneratorRunner()
        self.cacher = DataclassCacher(self)

        assert self.tag != ""
        self.set_save_path("results")  # Default analysis path

    def output(self):
        """
        Output viewer. If cache exist, read result from cache value.
        Otherwise, execute (__call__) the module and return the value.
        """
        if self.cacher.check_cached():
            self.logger.info(f"Using cache: {self.cacher.cache_dir}")

            def generator_func():
                yield from self.cacher.load_cached()

            output = generator_func()
        else:
            self.logger.info("Cache not found.")
            self.logger.info(f"Using runner: {self.runner.__class__} type.")
            args = self.receive()  # Receive data from upstream
            assert (
                len(args) > 0
            ), "No data received from upstream. Generator-operator must receive other generators from upstream."
            output = self.runner(self.__call__, args)

            # Callback: After-run
            self.callback_after_run(output)

            # Plotting: Only happened when cache is not called
            if not self.skip_plot:
                # TODO: Possible refactor in the future with operator/operator.py
                if len(args) == 0:
                    self.plot(output, None, show=False, save_path=True)
                elif len(args) == 1:
                    self.plot(output, args[0], show=False, save_path=True)
                else:
                    self.plot(output, args, show=False, save_path=True)
        return output
