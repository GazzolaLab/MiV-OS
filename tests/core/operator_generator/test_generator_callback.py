import os
from dataclasses import dataclass

import numpy as np
import pytest

from miv.core.datatype.pure_python import GeneratorType
from miv.core.operator_generator import GeneratorOperatorMixin, cache_generator_call


@dataclass
class MockGeneratorOperatorModule(GeneratorOperatorMixin):
    tag: str = "mock operator"

    def __post_init__(self):
        super().__init__()

    @cache_generator_call
    def __call__(self, inputs):
        return inputs * 2

    def generator_plot_test1(
        self, output, inputs, show=False, save_path=None, index=-1
    ):
        savepath = os.path.join(save_path, f"gen_test_{index}.npy")
        # Save temporary file
        if save_path is not None:
            np.save(savepath, output)

    def firstiter_plot_test1(self, output, inputs, show=False, save_path=None):
        savepath = os.path.join(save_path, "firstiter_test_9.npy")
        # Save temporary file
        if save_path is not None:
            np.save(savepath, output)


def generator_plot_test_callback(
    self, output, inputs, show=False, save_path=None, index=-1
):
    savepath = os.path.join(save_path, f"gen_callback_{index}.npy")
    # Save temporary file
    if save_path is not None:
        np.save(savepath, output)


def firstiter_plot_test_callback(self, output, inputs, show=False, save_path=None):
    savepath = os.path.join(save_path, "firstiter_callback_13.npy")
    # Save temporary file
    if save_path is not None:
        np.save(savepath, output)


def test_callback_firstiter_plot_from_callbacks(tmp_path):
    gen = GeneratorType([1, 2, 3, 4, 5])
    mock_operator = MockGeneratorOperatorModule()
    mock_operator.set_save_path(tmp_path / "results")
    mock_operator << generator_plot_test_callback
    mock_operator << firstiter_plot_test_callback

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
        # Callback defined
        expected_file = os.path.join(
            tmp_path.as_posix(),
            "results",
            mock_operator.analysis_path,
            f"gen_callback_{i}.npy",
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
    # Callback defined
    expected_file = os.path.join(
        tmp_path.as_posix(),
        "results",
        mock_operator.analysis_path,
        "firstiter_callback_13.npy",
    )
    assert os.path.exists(expected_file)
