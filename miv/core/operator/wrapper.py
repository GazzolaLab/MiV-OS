__doc__ = """

Useful wrapper functions for MIV operators.

.. autofunction:: miv.core.operator.wrapper.cache_call
.. autofunction:: miv.core.source.wrapper.cached_method
.. autofunction:: miv.core.operator_generator.wrapper.cache_generator_call

"""

from typing import Any, TypeVar, cast
from collections.abc import Callable

from .protocol import _Cachable

F = TypeVar("F")


def cache_call(func: Callable[..., F]) -> Callable[..., F]:
    """
    Cache the methods of the operator.
    Save the cache in the cacher object.
    """

    def wrapper(self: _Cachable, *args: Any, **kwargs: Any) -> F:
        tag = "data"
        cacher = self.cacher

        result = func(self, *args, **kwargs)
        if result is None:
            # In case the module does not return anything
            return cast(F, None)
        cacher.save_cache(result, tag=tag)
        cacher.save_config(tag=tag)
        return result

    return wrapper
