__doc__ = """

Useful wrapper functions for MIV operators.

.. autofunction:: miv.core.wrapper.wrap_cacher

.. autofunction:: miv.core.wrapper.cache_functional

"""

__all__ = [
    "wrap_cacher",
    "wrap_cacher_generator",
    "cache_functional",
]

import types
from typing import Protocol, Union

import functools
import inspect
from collections import UserList
from dataclasses import dataclass, make_dataclass

from miv.core.datatype import DataTypes, Extendable
from miv.core.operator.cachable import (
    DataclassCacher,
    FunctionalCacher,
    _CacherProtocol,
)
from miv.core.operator.operator import Operator, OperatorMixin


def wrap_cacher(func):
    """
    Cache the methods of the operator.
    Save the cache in the cacher object.
    """

    def wrapper(self: Operator, *args, **kwargs):
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


def wrap_cacher_generator(func):
    """
    Cache the methods of the operator.
    It is special case for the generator in-out stream.
    Save the cache in the cacher object with appropriate tag.

    If inputs are not all generators, it will run regular function.
    """

    def wrapper(self: Operator, *args, **kwargs):
        is_all_generator = all(inspect.isgenerator(v) for v in args) and all(
            inspect.isgenerator(v) for v in kwargs.values()
        )

        tag = "data"
        cacher: DataclassCacher = self.cacher

        if is_all_generator:

            def generator_func(*args):
                for idx, zip_arg in enumerate(zip(*args)):
                    result = func(self, *zip_arg, **kwargs)
                    if result is not None:
                        # In case the module does not return anything
                        cacher.save_cache(result, idx, tag=tag)
                    yield result
                else:
                    cacher.save_config(tag=tag)

            generator = generator_func(*args, *kwargs.values())
            return generator
        else:
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
