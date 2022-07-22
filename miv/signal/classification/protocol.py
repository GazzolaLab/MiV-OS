__all__ = ["SpikeClassificationModelProtocol"]

from typing import Any, Dict, Protocol

import numpy as np


class SpikeClassificationModelProtocol(Protocol):
    """Behavior definitijon of all spike classification models."""

    def fit(self, *args, **kwargs) -> None:
        ...

    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        ...

    def compile(self, *args, **kwargs) -> None:
        ...

    def predict(self, *args, **kwargs) -> np.ndarray:
        ...
