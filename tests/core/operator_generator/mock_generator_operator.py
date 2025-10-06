import os
from dataclasses import dataclass

import numpy as np
import pytest

from miv.core.operator_generator.operator import GeneratorOperatorMixin
from miv.core.operator_generator.wrapper import cache_generator_call


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


def generator_plot_test_callback(output, inputs, show=False, save_path=None, index=-1):
    savepath = os.path.join(save_path, f"gen_callback_{index}.npy")
    # Save temporary file
    if save_path is not None:
        np.save(savepath, output)


def firstiter_plot_test_callback(output, inputs, show=False, save_path=None):
    savepath = os.path.join(save_path, "firstiter_callback_13.npy")
    # Save temporary file
    if save_path is not None:
        np.save(savepath, output)


def generator_plot_test_callback_as_method(
    self, output, inputs, show=False, save_path=None, index=-1
):
    savepath = os.path.join(save_path, f"gen_callback2_{index}.npy")
    # Save temporary file
    if save_path is not None:
        np.save(savepath, output)


def firstiter_plot_test_callback_as_method(
    self, output, inputs, show=False, save_path=None
):
    savepath = os.path.join(save_path, "firstiter_callback2_15.npy")
    # Save temporary file
    if save_path is not None:
        np.save(savepath, output)
