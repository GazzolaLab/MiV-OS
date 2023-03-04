__all__ = [
    "wrap_generator_to_generator",
]

import types
from typing import Protocol, Union

import functools
import inspect
from collections import UserList
from dataclasses import dataclass, make_dataclass

from miv.core.datatype import DataTypes, Extendable
from miv.core.operator.operator import Operator, OperatorMixin


def wrap_generator_to_generator(func):
    """
    If all input arguments are generator, then the output will be a generator.

    TODO: Fix to accomodate free functions
    Current implementation only works for class methods.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self: Operator = args[0]
        is_all_generator = all(inspect.isgenerator(v) for v in args[1:]) and all(
            inspect.isgenerator(v) for v in kwargs.values()
        )
        if is_all_generator:

            def generator_func(*args, **kwargs):
                if self.cacher.check_cached():
                    yield from self.cacher.load_cached()
                else:
                    for idx, zip_arg in enumerate(zip(*args)):
                        result = func(self, *zip_arg, **kwargs)
                        self.cacher.save_cache(result, idx)
                        yield result
                    else:
                        self.cacher.save_config()

            return generator_func(*(args[1:]), **kwargs)
        else:
            if self.cacher.check_cached():
                return next(self.cacher.load_cached())
            else:
                result = func(*args, **kwargs)
                self.cacher.save_cache(result)
                self.cacher.save_config()
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
                print(type(args[0]))
                return func(self, *args)

            def __post_init__(self):
                _LambdaClass.__init__(self, **params)
                OperatorMixin.__init__(self)

        LambdaClass.__name__ = name
        obj = LambdaClass(**params)
        return obj

    return decorator


def test_wrap_generator():
    @wrap_generator_to_generator
    def foo(x, y):
        return x + y

    def bar():
        yield 1
        yield 2
        yield 3

    # FIXME: See docstring for wrap_generator_to_generator
    # assert foo(1, 2) == 3
    # assert tuple(foo(bar(), bar())) == (2, 4, 6)

    class foo_class:
        @wrap_generator_to_generator
        def __call__(self, x, y):
            return x + y

        @wrap_generator_to_generator
        def other(self, x, y):
            return x + y

    a = foo_class()
    assert a.other(1, 2) == 3
    assert a(1, 2) == 3
    assert tuple(a.other(bar(), bar())) == (2, 4, 6)
    assert tuple(a(bar(), bar())) == (2, 4, 6)


def test_wrap_miv_function():
    @miv_function("testTag", a=1, b=2)
    def func(self, c):
        print("here")
        print(c)

    print(func)
    print(func.__dir__())
    print(func(1))


if __name__ == "__main__":
    # test_wrap_generator()
    # test_output_generator_collapse()
    test_wrap_miv_function()
