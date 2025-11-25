from __future__ import annotations

from typing import Any
from collections.abc import Callable, Generator
from abc import ABC, abstractmethod


class RunnerBase(ABC):
    """Abstract base class for runner policies that control execution strategies.

    Runner policies define how operators execute their functions, supporting different
    parallelism strategies such as sequential execution, MPI-based distributed execution,
    or multiprocessing. Each operator can be configured with a specific runner policy
    to control its execution behavior.

    Implementations should define:
    - How to execute the function
    - How to handle parallel/distributed execution (MPI, multiprocessing, etc.)
    - The execution order for scheduling purposes

    ``VanillaRunner`` is the default runner policy that should define the most generic
    execution strategy.

    """

    @abstractmethod
    def __call__(
        self,
        func: Callable,
        inputs: Any | None = None,
    ) -> Generator[Any] | Any:
        pass

    @abstractmethod
    def get_run_order(self) -> int:
        """
        The method determines the order of execution, useful for
        multiprocessing or MPI.
        """
