from __future__ import annotations

import itertools
from typing import Any
from collections.abc import Generator, Iterable

from ..cache_write import persist_cacher_result
from ..chainable import ChainingMixin
from ..operator.cachable import DataclassCacher
from ..operator.policy import RunnerBase, VanillaRunner
from .callback import GeneratorCallbackMixin
from .policy import StreamChunkAlignedGeneratorRunner, VanillaGeneratorRunner


class GeneratorOperatorMixin(ChainingMixin, GeneratorCallbackMixin):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.runner: RunnerBase = VanillaRunner()
        self.analysis_path = "analysis"

        super().__init__(*args, cacher=DataclassCacher(self), **kwargs)

        self.tag: str
        self.runner = VanillaGeneratorRunner(self)

    def __repr__(self) -> str:
        return self.tag

    def __str__(self) -> str:
        return self.tag

    def flow_blocked(self) -> bool:
        try:
            return self.cacher.check_cached(skip_log=True)
        except (AttributeError, FileNotFoundError):
            return False

    @staticmethod
    def _tee_upstream_args(
        args: list[Iterable[Any]],
    ) -> tuple[list[Iterable[Any]], list[Iterable[Any]]]:
        """Fork each upstream iterable so the runner and plot/callback zip stay aligned."""
        tees = [itertools.tee(a, 2) for a in args]
        return [t[0] for t in tees], [t[1] for t in tees]

    def _wrap_streaming_chunk_side_effects(
        self,
        gen: Generator[Any, None, None],
        upstream_args: list[Iterable[Any]],
    ) -> Generator[Any, None, None]:
        """Persist each chunk and run generator plot hooks (lockstep stream runners only)."""
        tasks = zip(*upstream_args, strict=True)

        for idx, (result, zip_arg) in enumerate(zip(gen, tasks, strict=True)):
            persist_cacher_result(self.cacher, result, chunk_index=idx, tag="data")
            self._callback_generator_plot(
                idx, result, zip_arg, save_path=self.analysis_path
            )
            if idx == 0:
                self._callback_firstiter_plot(
                    result, zip_arg, save_path=self.analysis_path
                )
            yield result

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
            args_for_runner, args_for_callbacks = self._tee_upstream_args(args)
            raw_output = self.runner(self.__call__, args_for_runner)
            output = self._wrap_streaming_chunk_side_effects(
                raw_output, args_for_callbacks
            )
            # after_run / plot_* run per chunk inside _wrap_streaming_chunk_side_effects
        return output
