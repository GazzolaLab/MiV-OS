"""
The test written for reproducing issue 315
"""

import os
from dataclasses import dataclass

import numpy as np
import pytest

from miv.core.datatype.signal import Signal
from miv.core.operator.operator import DataLoaderMixin, OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.core.operator_generator.operator import GeneratorOperatorMixin
from miv.core.operator_generator.wrapper import cache_generator_call
from miv.core.pipeline import Pipeline
from miv.core.utils.graph_sorting import topological_sort


class MockDataLoaderNode(DataLoaderMixin):
    tag: str = "test data loader"

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
class MockGeneratorOperator(GeneratorOperatorMixin):
    tag: str = "test generator operator"

    def __post_init__(self):
        super().__init__()
        self.call_count = 0

    @cache_generator_call
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

    @cache_call
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


@pytest.mark.mpi_xfail
def test_pipeline_run1(pipeline, tmp_path):
    execution_order = topological_sort(pipeline.nodes_to_run[0])
    pipeline.run(tmp_path, verbose=True)

    assert len(execution_order) == 5
    assert execution_order[0].called

    assert os.path.exists(execution_order[1].analysis_path)
    assert os.path.exists(execution_order[2].analysis_path)
    assert os.path.exists(execution_order[3].analysis_path)
    assert os.path.exists(execution_order[4].analysis_path)

    assert os.path.exists(execution_order[1].cacher.cache_dir)
    assert os.path.exists(execution_order[2].cacher.cache_dir)
    assert os.path.exists(execution_order[3].cacher.cache_dir)
    assert os.path.exists(execution_order[4].cacher.cache_dir)

    assert execution_order[0].called
    assert execution_order[1].call_count == 10
    assert execution_order[2].call_count == 10
    assert execution_order[3].called
