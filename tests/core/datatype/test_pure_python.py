import numpy as np
import pytest

from miv.core.datatype.pure_python import (
    GeneratorType,
    NumpyDType,
    PythonDataType,
    ValuesMixin,
)
from miv.core.pipeline import Pipeline


class MockFloat(ValuesMixin):
    @staticmethod
    def is_valid(value):
        return isinstance(value, float)


def test_mock_values_mixin_on_pipelines(tmp_path):
    v1 = MockFloat(1.0)
    v2 = MockFloat(2.0)
    v3 = MockFloat(3.0)
    v1 >> v2 >> v3
    Pipeline(v3).run(tmp_path)
    assert v1.output() == 1.0
    assert v2.output() == 2.0
    assert v3.output() == 3.0

    assert v1.run() == 1.0
    assert v2.run() == 2.0
    assert v3.run() == 3.0


def test_python_datatype():
    v1 = PythonDataType(1)
    v2 = PythonDataType(2)
    v3 = PythonDataType(3)
    v1 >> v2 >> v3
    assert v1.output() == 1
    assert v2.output() == 2
    assert v3.output() == 3

    assert v1.run() == 1
    assert v2.run() == 2
    assert v3.run() == 3

    assert v1.is_valid(1.0)
    assert v2.is_valid(1.0)
    assert v3.is_valid(1.0)


def test_numpy_datatype():
    v1 = NumpyDType(np.array([1]))
    v2 = NumpyDType(np.array([2]))
    v3 = NumpyDType(np.array([3]))
    v1 >> v2 >> v3
    assert np.array_equal(v1.output(), np.array([1]))
    assert np.array_equal(v2.output(), np.array([2]))
    assert np.array_equal(v3.output(), np.array([3]))

    assert np.array_equal(v1.run(), np.array([1]))
    assert np.array_equal(v2.run(), np.array([2]))
    assert np.array_equal(v3.run(), np.array([3]))

    assert v1.is_valid(np.array([1]))
    assert v2.is_valid(np.array([1]))
    assert v3.is_valid(np.array([1]))


def test_generator_type():
    v1 = GeneratorType([1, 2, 0])
    assert np.array_equal(list(v1.output()), np.array([1, 2, 0]))

    assert np.array_equal(list(v1.run()), np.array([1, 2, 0]))

    def f():
        yield from range(3)

    assert v1.is_valid(f())
