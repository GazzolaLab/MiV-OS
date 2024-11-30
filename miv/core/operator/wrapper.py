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
from typing import Any, Callable, Protocol, Type, TypeVar, Union

import functools
import inspect
from collections import UserList
from dataclasses import dataclass, make_dataclass

from miv.core.datatype import DataTypes, Extendable

from .cachable import DataclassCacher, FunctionalCacher, _Cachable, _CacherProtocol
from .operator import Operator, OperatorMixin

F = TypeVar("F", bound=Callable[..., Any])


def cache_call(func: F) -> F:
    """
    Cache the methods of the operator.
    Save the cache in the cacher object.
    """

    def wrapper(self: _Cachable, *args, **kwargs):
        tag = "data"
        cacher: DataclassCacher = self.cacher

        result = func(self, *args, **kwargs)
        if result is None:
            # In case the module does not return anything
            return None
        cacher.save_cache(result, tag=tag)
        cacher.save_config(tag=tag)
        return result

    return wrapper


def cache_functional(cache_tag=None):
    """
    Cache the functionals.
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            cacher: FunctionalCacher = self.cacher
            tag = "data" if cache_tag is None else cache_tag

            # TODO: check cache by parameters should be improved
            if cacher.check_cached(params=(args, kwargs), tag=tag):
                cacher.cache_called = True
                loader = cacher.load_cached(tag=tag)
                value = next(loader)
                return value
            else:
                result = func(self, *args, **kwargs)
                if result is None:
                    return None
                cacher.save_cache(result, tag=tag)
                cacher.save_config(params=(args, kwargs), tag=tag)
                cacher.cache_called = False
                return result

        return wrapper

    return decorator
