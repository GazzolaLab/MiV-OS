import multiprocessing

import pytest

from miv.core.operator.policy import (
    RunnerBase,
    StrictMPIRunner,
    SupportMPIMerge,
    VanillaRunner,
)


def test_vanilla_runner_get_run_order():
    """
    VanillaRunner should implement get_run_order() method and return 0.
    """
    vr = VanillaRunner()
    assert hasattr(vr, "get_run_order"), (
        "VanillaRunner should have get_run_order method"
    )
    assert callable(getattr(vr, "get_run_order")), "get_run_order should be callable"
    assert vr.get_run_order() == 0, "VanillaRunner.get_run_order() should return 0"


def test_vanilla_runner_execute_function_with_no_inputs():
    """
    VanillaRunner should execute function with no inputs.
    """
    vr = VanillaRunner()
    execution_count = 0

    def func():
        nonlocal execution_count
        execution_count += 1
        return 42

    output = vr(func, inputs=None)

    assert output == 42
    assert execution_count == 1, "Function should execute exactly once"


def test_vanilla_runner_execute_function_with_single_argument():
    """
    VanillaRunner should execute function with inputs (single argument).
    """
    vr = VanillaRunner()
    execution_count = 0

    def func(x):
        nonlocal execution_count
        execution_count += 1
        return x * 2

    output = vr(func, inputs=(10,))

    assert output == 20
    assert execution_count == 1, "Function should execute exactly once"


@pytest.mark.parametrize(
    "inputs, expected_output", [((2, 3), 6), ((4, 5), 20), ((6, 7), 42)]
)
def test_vanilla_runner_execute_function_with_multiple_inputs(inputs, expected_output):
    """
    VanillaRunner should execute function with multiple inputs (tuple).
    """
    vr = VanillaRunner()
    execution_count = 0

    def func(x, y):
        nonlocal execution_count
        execution_count += 1
        return x * y

    output = vr(func, inputs=inputs)

    assert output == expected_output, f"Expected {expected_output}, but got {output}"
    assert execution_count == 1, "Function should execute exactly once"


def test_vanilla_runner_returns_result_directly():
    """
    VanillaRunner should return result directly (no parallel processing).
    """
    vr = VanillaRunner()

    def func(x):
        return x * 2

    output = vr(func, inputs=(5,))

    # Should return the result directly, not a generator or future
    assert output == 10
    assert not hasattr(output, "__iter__") or not hasattr(output, "__next__"), (
        "Result should be returned directly, not as a generator"
    )


def test_vanilla_runner_executes_only_once():
    """
    VanillaRunner should execute only once (sequential, not parallel).
    """
    vr = VanillaRunner()
    execution_count = 0

    def func(x):
        nonlocal execution_count
        execution_count += 1
        return x * 2

    # Call multiple times to verify sequential execution
    result1 = vr(func, inputs=(1,))
    result2 = vr(func, inputs=(2,))
    result3 = vr(func, inputs=(3,))

    assert result1 == 2
    assert result2 == 4
    assert result3 == 6
    assert execution_count == 3, (
        "Function should execute exactly 3 times (once per call)"
    )
