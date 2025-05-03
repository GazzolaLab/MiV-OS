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
        if not all_generator:
            result = func(self, *inputs, idx=0)
            return result

        def generator_func(*args: tuple[Generator, ...]) -> Generator:
            chunk_size = 8
            istart = 0
            while zip_arg := list(islice(args, chunk_size)):
                func = functools.partial(func, self)
                args = [tuple(arg + [istart + i]) for i, arg in enumerate(zip_arg)]
                with mp.Pool(processes=chunk_size) as pool:
                    results = pool.starmap(func, args)
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

        generator = generator_func(*args)
        return generator

        # return func(*inputs)  # type: ignore

    def get_run_order(self) -> int:
        return 0
