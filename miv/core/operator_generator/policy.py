__all__ = [
    "VanillaGeneratorRunner",
]

from typing import Any, TypeVar, TYPE_CHECKING, Protocol
from itertools import islice
import time
from collections.abc import Generator, Iterable

import multiprocessing as mp

from ..policy import RunnerBase

if TYPE_CHECKING:
    from .operator import GeneratorOperatorMixin


# Lazy-callable function type for generator operators.
# _LazyCallable = Callable[[int, Iterable[Any], ...], Any]
class _LazyCallable(Protocol):
    __name__: str

    def __call__(self, idx: int, *args: Iterable[Any]) -> Any: ...


C = TypeVar("C", bound="GeneratorOperatorMixin")


class VanillaGeneratorRunner(RunnerBase):
    """Default runner without any modification.
    Simply, the operator will be executed in embarrassingly parallel manner.
    If MPI is available, the operator will be executed in every ranks.
    (This is different behavior from regular operator VanillaRunner, although
    it would not create overhead due to the lazy execution strategy.)

    This runner is meant to be used for generator operators.
    """

    def __init__(self, parent: C) -> None:
        self.parent = parent
        self.logger = parent.logger

    def __call__(
        self,
        func: "_LazyCallable",
        inputs: list[Iterable[Any]] | None = None,
    ) -> Generator[Any, None, None]:
        if inputs is None:

            def _generator_func() -> Generator[Any, None, None]:
                yield func(0, *[])

            return _generator_func()
        else:
            if not isinstance(inputs, Iterable):
                raise ValueError("All inputs must be iterables")

            is_all_iterable = all(isinstance(v, Iterable) for v in inputs)
            if not is_all_iterable:
                raise ValueError("All inputs must be iterables")

        def generator_func(
            *args: Iterable[Any],
        ) -> Generator[Any, None, None]:
            tasks = zip(*args, strict=False)
            for idx, zip_arg in enumerate(tasks):
                stime = time.time()
                result = func(idx, *zip_arg)
                self.logger.info(
                    f"    iter {idx:03d} {self}: {time.time() - stime:.03f}sec",
                )
                yield result
            self.logger.info("generator-tasks done")

            # TODO: add lastiter_plot
            # FIXME
            self.parent._done_flag_generator_plot = True
            self.parent._done_flag_firstiter_plot = True

        generator = generator_func(*inputs)
        return generator

    def get_run_order(self) -> int:
        return 0


class GeneratorRunnerInMultiprocessing(RunnerBase):
    """Runner for generator operators using multiprocessing."""

    def __init__(self, parent: C, *, chunk_size: int = 4) -> None:
        self.parent = parent
        self.chunk_size = chunk_size

    def __call__(
        self,
        func: "_LazyCallable",
        inputs: list[Iterable[Any]] | None = None,
    ) -> Generator[Any, None, None]:
        logger = self.parent.logger
        if inputs is None:

            def _generator_func() -> Generator[Any, None, None]:
                yield func(0, *[])

            return _generator_func()
        else:
            if not isinstance(inputs, Iterable):
                raise ValueError("All inputs must be iterables")

            is_all_iterable = all(isinstance(v, Iterable) for v in inputs)
            if not is_all_iterable:
                raise ValueError("All inputs must be iterables")

        def generator_func(
            *args: Iterable[Any],
        ) -> Generator[Any, None, None]:
            num_workers = self.chunk_size
            istart = 0
            tasks = zip(*args, strict=False)
            while zip_arg := list(islice(tasks, num_workers)):
                proxy_func = getattr(self.parent.__class__, func.__name__)
                stime = time.time()
                _args = [
                    tuple([self.parent] + [istart + i] + list(za))
                    for i, za in enumerate(zip_arg)
                ]
                with mp.Pool(processes=num_workers) as pool:
                    # results = pool.imap(prox_func, _args)
                    results = pool.starmap(proxy_func, _args)
                istart += len(_args)
                logger.info(
                    f"completed tasks: {istart}(+{len(_args)}) ({time.time() - stime:.2f}sec)",
                )
                _args = []

                stime = time.time()
                yield from results

                logger.info(
                    f"external_tasks:  ({time.time() - stime:.2f}sec)",
                )

            logger.info("generator-tasks done")

            # for idx, zip_arg in enumerate(zip(*args, strict=False)):
            #    stime = time.time()
            #    result = func(self, *zip_arg, idx=idx)
            # print(f"    iter {idx:03d} {self}: {time.time() - stime:.03f}sec")
            #    yield result
            # TODO: add lastiter_plot
            # FIXME
            self.parent._done_flag_generator_plot = True
            self.parent._done_flag_firstiter_plot = True

        generator = generator_func(*inputs)
        return generator

    def get_run_order(self) -> int:
        return 0
