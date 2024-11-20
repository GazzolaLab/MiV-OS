import pytest

from miv.core.pipeline import Pipeline
from tests.core.mock_chain import MockChainRunnable, MockChainRunnableWithCache


def test_pipeline():
    pipeline = Pipeline(MockChainRunnable(1))
    assert pipeline is not None


@pytest.fixture
def pipeline():
    a = MockChainRunnable(1)
    b = MockChainRunnable(2)
    c = MockChainRunnable(3)
    d = MockChainRunnable(4)
    e = MockChainRunnable(5)
    a >> b >> d
    a >> c >> e >> d
    return Pipeline(e)


def test_pipeline_run(tmp_path, pipeline):
    pipeline.run(tmp_path / "results", verbose=True)


def test_pipeline_summarize(pipeline):
    pipeline.summarize()


def test_pipeline_execution_count(tmp_path):
    a = MockChainRunnable(1)
    b = MockChainRunnable(2)
    c = MockChainRunnable(3)

    Pipeline(c).run(tmp_path / "results")
    assert c.run_counter == 1

    Pipeline([a, c]).run(tmp_path / "results")
    assert a.run_counter == 1
    assert c.run_counter == 2