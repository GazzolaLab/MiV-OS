__all__ = [
    "VanillaGeneratorRunner",
]

from typing import Any
import inspect
import functools
from itertools import islice
from collections.abc import Generator

import multiprocessing as mp

from ..protocol import _LazyCallable


class VanillaGeneratorRunner:
    """Default runner without any modification.
    Simply, the operator will be executed in embarrassingly parallel manner.
    If MPI is available, the operator will be executed in every ranks.

    This runner is meant to be used for generator operators.
    """

    def __init__(self, parent):
        self.parent = parent

    def __call__(
        self, func: _LazyCallable, inputs: list[Generator[Any]] | None = None
    ) -> Generator:
        is_all_generator = all(inspect.isgenerator(v) for v in inputs)
        if not is_all_generator:
            result = func(self, *inputs)
            return result

        def generator_func(*args: tuple[Generator, ...]) -> Generator:
            chunk_size = 4
            istart = 0
            tasks = zip(*args)
            while zip_arg := list(islice(tasks, chunk_size)):
                proxy_func = getattr(self.parent.__class__, func.__name__)
                _args = [
                    tuple([self.parent] + [istart + i] + list(za))
                    for i, za in enumerate(zip_arg)
                ]
                with mp.Pool(processes=chunk_size) as pool:
                    results = pool.starmap(proxy_func, _args)
                istart += chunk_size

                for result in results:
                    yield result

            # for idx, zip_arg in enumerate(zip(*args, strict=False)):
            #    stime = time.time()
            #    result = func(self, *zip_arg, idx=idx)
            # print(f"    iter {idx:03d} {self}: {time.time() - stime:.03f}sec", flush=True)
            #    yield result
            # TODO: add lastiter_plot
            # FIXME
            self.parent._done_flag_generator_plot = True
            self.parent._done_flag_firstiter_plot = True

        generator = generator_func(*inputs)
        return generator

        # return func(*inputs)  # type: ignore

    def get_run_order(self) -> int:
        return 0
