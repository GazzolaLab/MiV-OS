__all__ = [
    "VanillaGeneratorRunner",
]

from typing import Any
import inspect
from itertools import islice
import time
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
            result = func(self, *inputs)  # FIXME: this will probably cause issue
            return result

        def generator_func(*args: tuple[Generator, ...]) -> Generator:
            num_workers = 1
            tasks = zip(*args, strict=False)
            for idx, zip_arg in enumerate(tasks):
                stime = time.time()
                result = func(idx, *zip_arg)
                print(f"    iter {idx:03d} {self}: {time.time() - stime:.03f}sec", flush=True)
                yield result
            print("generator-tasks done", flush=True)

            # TODO: add lastiter_plot
            # FIXME
            self.parent._done_flag_generator_plot = True
            self.parent._done_flag_firstiter_plot = True

        generator = generator_func(*inputs)
        return generator

        # return func(*inputs)  # type: ignore

    def get_run_order(self) -> int:
        return 0

class GeneratorRunnerInMultiprocessing:
    """Default runner without any modification.
    Simply, the operator will be executed in embarrassingly parallel manner.
    If MPI is available, the operator will be executed in every ranks.

    This runner is meant to be used for generator operators.
    """

    def __init__(self, parent, chunk_size:int=4):
        self.parent = parent
        self.chunk_size = chunk_size

    def __call__(
        self, func: _LazyCallable, inputs: list[Generator[Any]] | None = None
    ) -> Generator:
        is_all_generator = all(inspect.isgenerator(v) for v in inputs)
        if not is_all_generator:
            result = func(self, *inputs)  # FIXME: this will probably cause issue
            return result

        def generator_func(*args: tuple[Generator, ...]) -> Generator:
            num_workers = self.chunk_size
            istart = 0
            tasks = zip(*args, strict=False)
            with mp.Pool(processes=num_workers) as pool:
                while zip_arg := list(islice(tasks, num_workers)):
                    proxy_func = getattr(self.parent.__class__, func.__name__)
                    stime = time.time()
                    _args = [
                        tuple([self.parent] + [istart + i] + list(za))
                        for i, za in enumerate(zip_arg)
                    ]
                    # results = pool.imap(prox_func, _args)
                    results = pool.starmap(proxy_func, _args)
                    istart += len(_args)
                    print(
                        f"completed tasks: {istart}(+{len(_args)}) ({time.time() - stime:.2f}sec)",
                        flush=True,
                    )

                    stime = time.time()
                    yield from results
                    print(
                        f"external_tasks:  ({time.time() - stime:.2f}sec)", flush=True
                    )

            print("generator-tasks done", flush=True)

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
