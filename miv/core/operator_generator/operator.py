from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Generator


if TYPE_CHECKING:
    from ..operator.policy import RunnerBase

from ..operator.cachable import CACHE_POLICY
from ..operator.operator import OperatorMixin
from .callback import (
    GeneratorCallbackMixin,
)
from .policy import VanillaGeneratorRunner


class GeneratorOperatorMixin(OperatorMixin, GeneratorCallbackMixin):
    def __init__(self) -> None:
        super().__init__()

        self.runner: RunnerBase = VanillaGeneratorRunner(self)  # type: ignore[assignment]

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
