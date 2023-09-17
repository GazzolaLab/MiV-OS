import multiprocessing

import pytest

from miv.core.operator.policy import MultiprocessingRunner, VanillaRunner


@pytest.mark.parametrize(
    "inputs, expected_output", [((2, 3), 6), ((4, 5), 20), ((6, 7), 42)]
)
def test_vanilla_runner(inputs, expected_output):
    vr = VanillaRunner()

    def func(x, y):
        return x * y

    output = vr(func, inputs=inputs)

    assert output == expected_output, f"Expected {expected_output}, but got {output}"


@pytest.mark.parametrize(
    "n, expected_output", [(1, 1), (2, 2), (4, 4), (None, multiprocessing.cpu_count())]
)
def test_multiprocessing_number_of_processors(n, expected_output):
    mr = MultiprocessingRunner(np=n)
    assert (
        mr.num_proc == expected_output
    ), f"Expected {expected_output}, but got {mr.num_proc}"


def _func(x):
    return x * x


def test_multiprocessing_runner():
    mr = MultiprocessingRunner(np=2)

    inputs = [2, 4, 6, 8]
    expected_output = [4, 16, 36, 64]

    output = list(mr(_func, inputs=inputs))

    assert output == expected_output, f"Expected {expected_output}, but got {output}"


def test_multiprocessing_runner_no_input():
    mr = MultiprocessingRunner(np=2)

    with pytest.raises(NotImplementedError):
        next(mr(_func))

    with pytest.raises(NotImplementedError):
        next(mr(_func, None))
