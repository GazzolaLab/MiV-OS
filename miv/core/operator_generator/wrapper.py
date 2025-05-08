__all__ = [
    "cache_generator_call",
]

from typing import Any
from collections.abc import Generator, Callable

import time
import functools
import inspect

from .operator import GeneratorOperator


def cache_generator_call(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(
        self: GeneratorOperator,
        idx,
        *args,
    ) -> Generator | Any | None:
        tag = "data"
        cacher = self.cacher
        result = func(self, *args)
        if result is not None:
            cacher.save_cache(result, idx=idx, tag=tag)
            if idx == 0:
                cacher.save_config(tag=tag)  # config saved only once

        # Plotting  # FIXME: Somehow move it in operator mixin?
        self._callback_generator_plot(idx, result, args, save_path=self.analysis_path)
        if idx == 0:
            self._callback_firstiter_plot(result, args, save_path=self.analysis_path)

        return result

    return wrapper


# def cache_generator_call(func: Callable) -> Callable:
#    """
#    Cache the methods of the operator.
#    It is special case for the generator in-out stream.
#    Save the cache in the cacher object with appropriate tag.
#
#    If inputs are not all generators, it will run regular function.
#    """
#
#    def wrapper(
#        self: GeneratorOperator, *args: Any, **kwargs: Any
#    ) -> Generator | Any | None:
#        is_all_generator = all(inspect.isgenerator(v) for v in args) and all(
#            inspect.isgenerator(v) for v in kwargs.values()
#        )
#
#        tag = "data"
#        cacher = self.cacher
#
#        if is_all_generator:
#
#            def generator_func(*args: tuple[Generator, ...]) -> Generator:
#                for idx, zip_arg in enumerate(zip(*args, strict=False)):
#                    stime = time.time()
#                    result = func(self, *zip_arg, **kwargs)
#                    # print(f"    iter {idx:03d} {self}: {time.time() - stime:.03f}sec", flush=True)
#                    if result is not None:
#                        # In case the module does not return anything
#                        cacher.save_cache(result, idx, tag=tag)
#                    self._callback_generator_plot(
#                        idx, result, zip_arg, save_path=self.analysis_path
#                    )
#                    if idx == 0:
#                        self._callback_firstiter_plot(
#                            result, zip_arg, save_path=self.analysis_path
#                        )
#                    yield result
#                cacher.save_config(tag=tag)
#                # TODO: add lastiter_plot
#                # FIXME
#                self._done_flag_generator_plot = True
#                self._done_flag_firstiter_plot = True
#
#            generator = generator_func(*args, *kwargs.values())
#            return generator
#        else:
#            result = func(self, *args, **kwargs)
#            if result is None:
#                # In case the module does not return anything
#                return None
#            cacher.save_cache(result, tag=tag)
#            cacher.save_config(tag=tag)
#            return result
#
#    return wrapper
