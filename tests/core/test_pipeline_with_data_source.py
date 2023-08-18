"""
The test written for reproducing issue 315
"""

import os
from dataclasses import dataclass

import numpy as np
import pytest

from miv.core.datatype.signal import Signal
from miv.core.operator.operator import DataLoaderMixin, OperatorMixin
from miv.core.pipeline import Pipeline
from miv.core.wrapper import wrap_cacher, wrap_generator_to_generator


class MockDataLoaderNode(DataLoaderMixin):
    def __init__(self, path):
        self.data_path = path
        super().__init__()

        self.called = False

    def load(self, *args, **kwargs):
        assert not self.called
        self.called = True

        for _ in range(10):
            rate = 10
            T = 10
            timestamps = np.linspace(0, T, T * rate)[:, None]
            signal = np.sin(2 * np.pi * timestamps)
            yield Signal(data=signal, timestamps=timestamps, rate=rate)

    def check_path_validity(self):
        return True


@dataclass
class MockGeneratorOperator(OperatorMixin):
    tag: str = "test generator operator"

    def __post_init__(self):
        super().__init__()
        self.call_count = 0

    @wrap_generator_to_generator
    def __call__(self, data):
        self.call_count += 1
        assert self.call_count <= 10
        return data


@dataclass
class MockOperator(OperatorMixin):
    tag: str = "test operator"

    def __post_init__(self):
        super().__init__()
        self.called = False

    @wrap_cacher()
    def __call__(self, data):
        assert not self.called
        self.called = True

        results = list(data)
        return results


@pytest.fixture
def pipeline(tmp_path):
    dataloader = MockDataLoaderNode(path=tmp_path)
    gen_ops1 = MockGeneratorOperator(tag="test_generator_operator1")
    gen_ops2 = MockGeneratorOperator(tag="test_generator_operator2")
    ops1 = MockOperator(tag="test_operator1")
    ops2 = MockOperator(tag="test_operator2")

    dataloader >> gen_ops1 >> gen_ops2 >> ops1 >> ops2
    return Pipeline(ops2)


def test_pipeline_run1(pipeline, tmp_path):
    pipeline.run(tmp_path, verbose=True)

    assert len(pipeline.execution_order) == 5
    assert pipeline.execution_order[0].called
    assert pipeline.execution_order[1].call_count == 10
    assert pipeline.execution_order[2].call_count == 10
    assert pipeline.execution_order[3].called
    assert pipeline.execution_order[4].called

    assert os.path.exists(pipeline.execution_order[1].analysis_path)
    assert os.path.exists(pipeline.execution_order[2].analysis_path)
    assert os.path.exists(pipeline.execution_order[3].analysis_path)
    assert os.path.exists(pipeline.execution_order[4].analysis_path)

    assert os.path.exists(pipeline.execution_order[1].cacher.cache_dir)
    assert os.path.exists(pipeline.execution_order[2].cacher.cache_dir)
    assert os.path.exists(pipeline.execution_order[3].cacher.cache_dir)
    assert os.path.exists(pipeline.execution_order[4].cacher.cache_dir)
