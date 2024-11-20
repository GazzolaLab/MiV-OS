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
        self.runner = VanillaGeneratorRunner()
        self._cacher = DataclassCacher(self)
        super().__init__()

    @property
    def cacher(self) -> _CacherProtocol:
        return self._cacher

    @cacher.setter
    def cacher(self, value: _CacherProtocol) -> None:
        # FIXME:
        policy = self._cacher.policy
        cache_dir = self._cacher.cache_dir
        self._cacher = value(self)
        self._cacher.policy = policy
        self._cacher.cache_dir = cache_dir

    def set_caching_policy(self, policy: CACHE_POLICY) -> None:
        self.cacher.policy = policy

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
            self._callback_after_run(output)

            # Plotting: Only happened when cache is not called
            self._callback_plot(output, args, show=False, save_path=True)
        return output
