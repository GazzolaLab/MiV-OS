__doc__ = """

Useful wrapper functions for MIV operators.

.. autofunction:: miv.core.wrapper.cache_call

.. autofunction:: miv.core.wrapper.cached_method

.. autofunction:: miv.core.operator_generator.wrapper.cache_generator_call


"""

__all__ = [
    "cache_call",
    "cached_method",
]

from typing import Any, TypeVar, Concatenate, ParamSpec, cast
from collections.abc import Callable
import os
import functools
from joblib import Memory

from .protocol import _Cachable

F = TypeVar("F")
P = ParamSpec("P")
R = TypeVar("R")
SELF = TypeVar("SELF", bound=_Cachable)


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


def cached_method(
    cache_tag: str | None = None,
    *,
    verbose: bool = False,
) -> Callable[[Callable[Concatenate[SELF, P], R]], Callable[Concatenate[SELF, P], R]]:
    """
    Cache the results of instance methods.
    Uses joblib's Memory to handle caching with optional custom tag and verbosity.

    Example
    -------
        >>> from miv.core.operator.wrapper import cached_method
        >>> from miv.core.operator.protocol import _Cachable
        >>>
        >>> class MyOperator(OperatorMixin):
        ...     def __init__(self):
        ...         self.cacher: _CacherProtocol
        ...
        ...     @cached_method("my_tag")  # Cache with custom tag. Otherwise, it uses the function name.
        ...     def expensive_computation(self, x: int) -> int:
        ...         # This will only be computed once for each unique input
        ...         return x * x
        ...
        >>> op = MyOperator()
        >>> result = op.expensive_computation(5)  # Computes and caches
        >>> result = op.expensive_computation(5)  # Returns cached result
    """

    def decorator(
        func: Callable[Concatenate[SELF, P], R],
    ) -> Callable[Concatenate[SELF, P], R]:
        @functools.wraps(func)
        def wrapper(self: SELF, *args: P.args, **kwargs: P.kwargs) -> R:
            # Check if caching is allowed based on policy
            policy = self.cacher.policy
            if policy == "OFF":
                # Never use cache, always run the function
                return func(self, *args, **kwargs)

            # For other policies, we'll use the caching mechanism
            cached_methods = getattr(self, "_cached_methods", {})
            cache_dir = self.cacher.cache_dir
            tag = cache_tag if cache_tag is not None else func.__name__

            if tag not in cached_methods:
                cache_path = os.path.join(cache_dir, tag)
                memory = Memory(cache_path, verbose=verbose)
                _func = memory.cache(
                    func, ignore=["self"]
                )  # ignore is used to bypass un-hashable self
                cached_methods[tag] = _func
                setattr(self, "_cached_methods", cached_methods)  # noqa
            if policy == "MUST":
                if not cached_methods[tag].check_call_in_cache(*args, **kwargs):  # type: ignore
                    raise RuntimeError(
                        f"MUST policy is used for caching, but cache for {tag} does not exist."
                    )
            elif policy == "OVERWRITE":
                cached_methods[tag].clear(warn=False)  # type: ignore

            return cached_methods[tag](self, *args, **kwargs)  # type: ignore

        return wrapper  # type: ignore

    return decorator
