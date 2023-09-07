import logging

import pytest

from miv.utils.progress_logging import pbar


@pytest.mark.parametrize(
    "iterable, step",
    [
        (range(100), 25),
        (range(100), 10),
        (range(100), 5),
        (range(100), 1),
    ],
)
def test_pbar(iterable, step, caplog):
    logger = logging.getLogger(__name__)
    for i in pbar(iterable, logger, step=step):
        pass
    for log in caplog.record_tuples:
        assert log[2].startswith("complete: ")
        assert int(log[2].split(" ")[1][:-1]) % step == 0


@pytest.mark.parametrize(
    "iterable, step",
    [
        (range(100), 0),
        (range(100), -1),
        (range(100), 101),
        (range(100), 1000),
    ],
)
def test_pbar_wrong_step(iterable, step):
    logger = logging.getLogger(__name__)
    with pytest.raises(AssertionError):
        for i in pbar(iterable, logger, step=step):
            pass
