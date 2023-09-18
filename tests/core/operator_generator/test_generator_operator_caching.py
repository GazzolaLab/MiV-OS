import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from miv.core.datatype.pure_python import GeneratorType
from tests.core.operator_generator.mock_generator_operator import (
    MockGeneratorOperatorModule,
)


@patch.object(MockGeneratorOperatorModule, "generator_plot_test1")
@patch.object(MockGeneratorOperatorModule, "firstiter_plot_test1")
def test_mock_operator_caching(
    mock_firstiter_plot_test1, mock_generator_plot_test1, tmp_path
):
    gen = GeneratorType([1, 2, 3, 4, 5])
    mock_operator = MockGeneratorOperatorModule()
    mock_operator.set_save_path(tmp_path / "results")
    mock_operator.cacher.policy = "ON"

    gen >> mock_operator
    results = list(mock_operator.output())

    # print tmp_path / "results" recursively
    start_path = tmp_path / "results" / "mock_operator"
    print(os.listdir(start_path))
    cache_path = tmp_path / "results" / "mock_operator" / ".cache"
    print(os.listdir(cache_path))

    expected = [2, 4, 6, 8, 10]
    assert results == expected
    # generator_plotter1 should be called 5 times
    assert mock_generator_plot_test1.call_count == 5
    # firstiter_plot_test1 should be called 1 time
    assert mock_firstiter_plot_test1.call_count == 1

    # reset call count
    mock_generator_plot_test1.reset_mock()
    mock_firstiter_plot_test1.reset_mock()

    results = list(mock_operator.output())
    assert results == expected
    # When cache value is used, plotting should not be executed
    assert mock_generator_plot_test1.call_count == 0
    assert mock_firstiter_plot_test1.call_count == 0
