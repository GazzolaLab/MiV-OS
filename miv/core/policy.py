__all__ = ["_Runnable", "_RunnerProtocol", "VanillaRunner"]

from typing import Callable, Protocol


class _RunnerProtocol(Callable, Protocol):
    ...


class _Runnable(Protocol):
    """
    A protocol for a runner policy.
    """

    @property
    def runner(self) -> _RunnerProtocol:
        ...

    def run(self):
        ...


class VanillaRunner:
    """Default runner without any high-level parallelism."""

    def __call__(self, func, inputs, **kwargs):
        output = func(*inputs)
        return output


class StrictMPI:
    pass


class SupportMPI(StrictMPI):
    pass


class SupportMultiprocessing:
    pass
