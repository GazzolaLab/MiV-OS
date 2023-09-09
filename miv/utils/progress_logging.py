def pbar(iterable, logger, step=25):
    """
    Every step%, log the progress of the iterable

    How to use:
    >>> from miv.utils.progress_logging import pbar
    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> for i in pbar(range(100), logger, step=25):
    ...     pass
    complete: 25%
    complete: 50%
    complete: 75%
    complete: 100%
    """
    assert 0 < step <= 100, "step must be in (0, 100]"
    total = len(iterable)
    for i, item in enumerate(iterable):
        yield item
        if (i + 1) % (total // step) == 0:
            logger.info(f"complete: {((i + 1) / total) * 100:.0f}%")
    logger.info("complete: 100%")
