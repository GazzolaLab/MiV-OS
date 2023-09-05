__doc__ = """

Useful wrapper functions for MIV operators.

.. autofunction:: miv.core.wrapper.wrap_cacher

.. autofunction:: miv.core.wrapper.wrap_generator_to_generator

"""

__all__ = [
    "wrap",
    "wrap_cacher",
    "wrap_generator_to_generator",
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


def wrap(func):
    def wrapper(self, *args, **kwargs):
        inputs = self.receive()
        result = func(self, *inputs)
        if result is None:
            return None
        return result

    return wrapper


def wrap_cacher(cache_tag=None):
    """
    Decorator to wrap the function to use cacher.

    Note: Keep cache_tag=None for __call__ function.
    """

    def decorator(func):
        # @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self: Operator = args[0]
            cacher: _CacherProtocol = self.cacher
            tag = "data" if cache_tag is None else cache_tag

            if cacher.check_cached(params=(args[1:], kwargs), tag=tag):
                cacher.cache_called = True
                loader = cacher.load_cached(tag=tag)
                value = next(loader)
                return value
            else:
                # TODO: Need to clean this part
                if isinstance(cacher, FunctionalCacher):
                    result = func(*args, **kwargs)
                elif isinstance(cacher, DataclassCacher):
                    inputs = self.receive()
                    result = func(self, *inputs)
                if result is None:
                    return None
                cacher.save_cache(result, tag=tag)
                cacher.save_config(params=(args[1:], kwargs), tag=tag)
                cacher.cache_called = False
                return result

        return wrapper

    return decorator


def wrap_generator_to_generator(func):
    """
    If all input arguments are generator, then the output will be a generator.

    TODO: Fix to accomodate free functions
    Current implementation only works for class methods.
    """

    # @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        is_all_generator = all(inspect.isgenerator(v) for v in args) and all(
            inspect.isgenerator(v) for v in kwargs.values()
        )
        if is_all_generator:
            # TODO: Temporary fix
            if isinstance(self.cacher, DataclassCacher):
                args = self.receive()

            if self.cacher.check_cached():
                self.cacher.cache_called = True

                def generator_func(*args, **kwargs):
                    yield from self.cacher.load_cached()

            else:
                self.cacher.cache_called = False

                def generator_func(*args, **kwargs):
                    for idx, zip_arg in enumerate(zip(*args)):
                        result = func(self, *zip_arg, **kwargs)
                        if result is not None:
                            self.cacher.save_cache(result, idx)
                        yield result
                    else:
                        self.cacher.save_config()

            # return generator_func(*args, **kwargs)
            return generator_func(*args)
        else:
            if self.cacher.check_cached():
                self.cacher.cache_called = True
                return next(self.cacher.load_cached())
            else:
                result = func(self, *args, **kwargs)
                if result is None:
                    return None
                self.cacher.save_cache(result)
                self.cacher.save_config()
                self.cacher.cache_called = False
                return result

    return wrapper


def miv_function(name, **params):
    """
    Decorator to convert a function into a MIV operator.
    """

    def decorator(func):
        _LambdaClass = make_dataclass(
            name,
            [k for k in params.keys()],
        )

        @dataclass
        class LambdaClass(_LambdaClass, OperatorMixin):
            tag: str = name

            def __call__(self, *args, **kwargs):
                return func(self, *args)

            def __post_init__(self):
                _LambdaClass.__init__(self, **params)
                OperatorMixin.__init__(self)

        LambdaClass.__name__ = name
        obj = LambdaClass(**params)
        return obj

    return decorator


def test_wrap_miv_function():
    @miv_function("testTag", a=1, b=2)
    def func(self, c):
        print("here")
        print(c)

    print(func)
    print(func.__dir__())
    print(func(1))


if __name__ == "__main__":
    test_wrap_miv_function()
