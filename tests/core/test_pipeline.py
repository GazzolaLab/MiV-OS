import pytest
from mock_chain import MockChainRunnable

from miv.core.pipeline import Pipeline


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


def test_pipeline_run(pipeline):
    pipeline.run(verbose=True)


def test_pipeline_summarize(pipeline):
    pipeline.summarize()
