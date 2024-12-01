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

import types
from typing import Any, Protocol, Type, TypeVar, Union
from collections.abc import Callable

import functools
import inspect
from collections import UserList
from dataclasses import dataclass, make_dataclass

from miv.core.datatype import DataTypes, Extendable

from .cachable import DataclassCacher, FunctionalCacher, _CacherProtocol
from .operator import Operator, OperatorMixin, _Cachable

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


def cache_functional(
    cache_tag: str | None = None,
) -> Callable[[Callable[..., F]], Callable[..., F]]:
    """
    Cache the functionals.
    """

    def decorator(func: Callable[..., F]) -> Callable[..., F]:
        def wrapper(self: _Cachable, *args: Any, **kwargs: Any) -> F:
            cacher = self.cacher
            tag = "data" if cache_tag is None else cache_tag

            # TODO: check cache by parameters should be improved
            if cacher.check_cached(params=(args, kwargs), tag=tag):
                loader = cacher.load_cached(tag=tag)
                value = next(loader)
                return value  # type: ignore[no-any-return]
            else:
                result = func(self, *args, **kwargs)
                if result is None:
                    return None
                cacher.save_cache(result, tag=tag)
                cacher.save_config(params=(args, kwargs), tag=tag)
                return result

        return wrapper

    return decorator
