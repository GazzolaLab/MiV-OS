__doc__ = """

Useful wrapper functions for MIV operators.

.. autofunction:: miv.core.wrapper.cache_call

.. autofunction:: miv.core.wrapper.cache_functional

.. autofunction:: miv.core.operator_generator.wrapper.cache_generator_call


"""

__all__ = [
    "cache_call",
    "cache_functional",
]

from typing import Any, TypeVar, Concatenate, ParamSpec
from collections.abc import Callable
import os
import functools
from joblib import Memory

from .protocol import _Cachable

F = TypeVar("F")


def cache_call(func: Callable[..., F]) -> Callable[..., F]:
    """
    Cache the methods of the operator.
    Save the cache in the cacher object.
    """

    def wrapper(self: _Cachable, *args: Any, **kwargs: Any) -> F:
        tag = "data"
        cacher = self.cacher

        result = func(self, *args, **kwargs)
        if result is None:
            # In case the module does not return anything
            return None
        cacher.save_cache(result, tag=tag)
        cacher.save_config(tag=tag)
        return result

    return wrapper


SELF = TypeVar("SELF", bound=_Cachable)
P = ParamSpec("P")
R = TypeVar("R")


def cache_functional(
    cache_tag: str | None = None,
    verbose: bool = False,
) -> Callable[[Callable[Concatenate[SELF, P], R]], Callable[Concatenate[SELF, P], R]]:
    """
    Cache the functionals.
    """

    def decorator(
        func: Callable[Concatenate[SELF, P], R],
    ) -> Callable[Concatenate[SELF, P], R]:
        @functools.wraps(func)
        def wrapper(self: _Cachable, *args: Any, **kwargs: Any) -> R:
            cached_methods: dict[str, Callable] = getattr(self, "_cached_methods", {})
            cache_dir = self.cacher.cache_dir
            tag = cache_tag if cache_tag is not None else func.__name__
            if tag not in cached_methods:
                cache_path = os.path.join(cache_dir, tag)
                memory = Memory(cache_path, verbose=verbose)
                # Dev note: __get__ is used to bind the method to this "self" instance.
                # This is necessary because the memory.cache decorator expects a bound method.
                # Otherwise, there is no way to pass the "self" instance during the cache call.
                _func = memory.cache(func.__get__(self, type(self)))
                cached_methods[tag] = _func
            return cached_methods[tag](*args, **kwargs)

        return wrapper

    return decorator
