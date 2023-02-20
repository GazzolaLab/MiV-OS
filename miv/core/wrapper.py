import types
from typing import Union

import functools
import inspect
from collections import UserList

from miv.core.datatype import DataTypes, Extendable


def wrap_generator_to_generator(func):
    """
    If all input arguments are generator, then the output will be a generator.

    TODO: Fix to accomodate free functions
    Current implementation only works for class methods.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        is_all_generator = all(inspect.isgenerator(arg) for arg in args[1:])
        if is_all_generator:

            def generator_func(*args, **kwargs):
                for zip_arg in zip(*args):
                    yield func(self, *zip_arg, **kwargs)

            return generator_func(*(args[1:]), **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


def wrap_output_generator_collapse(*datatype_args: Union[DataTypes, Extendable]):
    """
    If all input arguments are generator, then the Generator output will be collapsed
    into designated datatypes.

    TODO: Fix to accomodate free functions
    TODO: Allow for mixture of Generators and non-Generators returns
    """

    def wrapper_gen(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
            is_output_generator = inspect.isgenerator(results)
            if is_output_generator:  # Collapse
                stacks = [T() for T in datatype_args]
                for result in results:
                    if not isinstance(result, tuple):
                        result = (result,)
                    assert len(result) == len(datatype_args)
                    for idx, item in enumerate(result):
                        stacks[idx].extend(item)
                if len(stacks) == 1:
                    return stacks[0]
                else:
                    return tuple(stacks)
            else:
                return results

        return wrapper

    return wrapper_gen


def test_output_generator_collapse():
    def bar():
        yield 1
        yield 2
        yield 3

    class ExtendableList(UserList):
        def extend(self, other):
            self.data.append(other)

    class foo_class:
        @wrap_output_generator_collapse(ExtendableList, ExtendableList)
        @wrap_generator_to_generator
        def __call__(self, x, y):
            return x + 1, y

        @wrap_output_generator_collapse(ExtendableList)
        @wrap_generator_to_generator
        def other(self, x, y):
            return x + 1

    a = foo_class()
    assert a.other(1, 2) == 2
    assert a(1, 2) == (2, 2)
    assert a.other(bar(), bar()) == ([2, 3, 4])
    assert a(bar(), bar()) == ([2, 3, 4], [1, 2, 3])


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


if __name__ == "__main__":
    test_wrap_generator()
    test_output_generator_collapse()