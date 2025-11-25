"""Tests for generator operator runner policies."""

import pytest
import logging
from collections.abc import Generator, Iterator, Iterable
from typing import Any

from miv.core.operator_generator.policy import (
    VanillaGeneratorRunner,
    GeneratorRunnerInMultiprocessing,
)


class DummyParent:
    """Dummy parent object for testing runners."""

    logger = logging.getLogger("test")


generator_runner_list = [
    VanillaGeneratorRunner(DummyParent()),
    GeneratorRunnerInMultiprocessing(DummyParent(), chunk_size=2),
]


@pytest.mark.parametrize("runner", generator_runner_list)
def test_generator_runner_raises_error_when_inputs_not_all_iterables(runner):
    """
    GeneratorRunner should raise error when inputs are not all iterables.
    """

    def func(idx, *args):
        return args

    with pytest.raises(ValueError, match="All inputs must be iterables"):
        runner(func, inputs=[42])  # No iterable

    with pytest.raises(ValueError, match="All inputs must be iterables"):
        runner(func, inputs=42)  # No list

    with pytest.raises(ValueError, match="All inputs must be iterables"):
        runner(func, inputs=[None])  # None

    # Test with mixed inputs (one generator, one non-iterable)
    def gen():
        yield 1

    with pytest.raises(ValueError, match="All inputs must be iterables"):
        runner(func, inputs=[gen(), 42])  # Mixed iterable and non-iterable

    with pytest.raises(ValueError, match="All inputs must be iterables"):
        runner(
            func,
            inputs=[
                (
                    1,
                    2,
                    3,
                ),
                42,
            ],
        )  # Mixed iterable and non-iterable


@pytest.mark.parametrize(
    "runner",
    generator_runner_list,
)
def test_generator_runner_accepts_iterables(runner):
    """
    GeneratorRunner should accept all iterables (generators, lists, iterators, tuples).
    """

    def func(idx, *args):
        for args in zip(*args, strict=False):
            yield args

    result = runner(func, inputs=[[1, 2, 3], [4, 5, 6]])
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")

    # Test that generators are accepted (generators are iterables)
    def gen1():
        yield 1
        yield 2

    def gen2():
        yield 2
        yield 4

    result = runner(func, inputs=[gen1(), gen2()])
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")

    # Test that iterators are accepted (iterators are iterables)
    iterator1 = iter([1, 2, 3])
    iterator2 = iter([4, 5, 6])
    result = runner(func, inputs=[iterator1, iterator2])
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")

    # Test that tuples are accepted (tuples are iterables)
    result = runner(func, inputs=[(1, 2, 3), (4, 5, 6)])
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")

    # Test that generators and iterators are accepted
    result = runner(func, inputs=[gen1(), iterator1])
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")


@pytest.mark.parametrize("runner", generator_runner_list)
def test_generator_runner_runs_function_without_inputs_when_inputs_is_none(runner):
    """
    GeneratorRunner should run the function without any inputs when inputs is None.
    """

    def func(idx, *args):
        return 42

    result = runner(func)
    # When inputs is None, runner returns a generator that yields the function result
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")
    # Consume the generator to get the actual result
    actual_result = next(result)
    assert actual_result == 42
