from __future__ import annotations

__all__ = [
    "GeneratorOperatorMixin",
]

from typing import TYPE_CHECKING, Protocol
from collections.abc import Generator


if TYPE_CHECKING:
    pass

from miv.core.operator.cachable import CACHE_POLICY
from miv.core.operator.operator import OperatorMixin
from miv.core.operator.protocol import _Node
from miv.core.operator_generator.callback import (
    GeneratorCallbackMixin,
    _GeneratorCallback,
)
from miv.core.operator_generator.policy import VanillaGeneratorRunner


class GeneratorOperator(
    _Node,
    _GeneratorCallback,
    Protocol,
): ...


class GeneratorOperatorMixin(OperatorMixin, GeneratorCallbackMixin):
    def __init__(self) -> None:
        super().__init__()

        self.runner = VanillaGeneratorRunner(self)

    def set_caching_policy(self, policy: CACHE_POLICY) -> None:
        self.cacher.policy = policy

    def output(self) -> Generator:
        """
        Output viewer. If cache exist, read result from cache value.
        Otherwise, execute (__call__) the module and return the value.
        """
        if self.cacher.check_cached():
            self.logger.info(f"Using cache: {self.cacher.cache_dir}")

            def generator_func() -> Generator:
                yield from self.cacher.load_cached()

            output = generator_func()
        else:
            self.logger.info("Cache not found.")
            self.logger.info(f"Using runner: {self.runner.__class__} type.")
            args = self.receive()  # Receive data from upstream
            assert len(args) > 0, (
                "No data received from upstream. Generator-operator must receive other generators from upstream."
            )
            output = self.runner(self.__call__, args)

            # Callback: After-run
            self._callback_after_run(output)

            # Plotting: Only happened when cache is not called
            self._callback_plot(output, args, show=False)
        return output
