import pytest


class MockRunner:
    """Default runner without any high-level parallelism."""

    def __call__(self, func, inputs=None, **kwargs):
        return True
