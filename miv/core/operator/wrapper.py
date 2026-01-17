__doc__ = """
"""

from collections.abc import Callable
import warnings
from typing import Any


def cache_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    DEPRECATED: This decorator is deprecated. No longer necessary.
    Output method will handle the cache saving logic.
    """

    warnings.warn(
        "This decorator is deprecated. No longer necessary.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return func
