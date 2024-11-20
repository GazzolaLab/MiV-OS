import os
from dataclasses import dataclass

import numpy as np
import pytest

from miv.core.datatype.pure_python import GeneratorType
from tests.core.operator_generator.mock_generator_operator import (
    MockGeneratorOperatorModule,
    firstiter_plot_test_callback,
    generator_plot_test_callback,
    generator_plot_test_callback_as_method,
    firstiter_plot_test_callback_as_method,
)


def test_callback_firstiter_plot_from_callbacks(tmp_path):
    gen = GeneratorType([1, 2, 3, 4, 5])
    mock_operator = MockGeneratorOperatorModule()
    mock_operator.set_save_path(tmp_path / "results")
    mock_operator << generator_plot_test_callback
    mock_operator << firstiter_plot_test_callback
    mock_operator << generator_plot_test_callback_as_method
    mock_operator << firstiter_plot_test_callback_as_method

    gen >> mock_operator
    results = list(mock_operator.output())
    expected = [2, 4, 6, 8, 10]
    assert results == expected

    # Generator files
    for i, result in enumerate(results):
        # In-class defined
        expected_file = os.path.join(
            tmp_path.as_posix(),
            "results",
            mock_operator.analysis_path,
            f"gen_test_{i}.npy",
        )
        assert os.path.exists(expected_file)
        # Callback defined (attribute)
        expected_file = os.path.join(
            tmp_path.as_posix(),
            "results",
            mock_operator.analysis_path,
            f"gen_callback_{i}.npy",
        )
        assert os.path.exists(expected_file)
        # Callback defined (instance method)
        expected_file = os.path.join(
            tmp_path.as_posix(),
            "results",
            mock_operator.analysis_path,
            f"gen_callback2_{i}.npy",
        )
        assert os.path.exists(expected_file)

    # First-iteration files
    # In-class defined
    expected_file = os.path.join(
        tmp_path.as_posix(),
        "results",
        mock_operator.analysis_path,
        "firstiter_test_9.npy",
    )
    assert os.path.exists(expected_file)
    # Callback defined (attribute)
    expected_file = os.path.join(
        tmp_path.as_posix(),
        "results",
        mock_operator.analysis_path,
        "firstiter_callback_13.npy",
    )
    assert os.path.exists(expected_file)
    # Callback defined (instance method)
    expected_file = os.path.join(
        tmp_path.as_posix(),
        "results",
        mock_operator.analysis_path,
        "firstiter_callback2_15.npy",
    )
    assert os.path.exists(expected_file)
