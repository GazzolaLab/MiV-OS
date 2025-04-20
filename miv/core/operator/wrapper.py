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

from typing import Any, TypeVar, Concatenate, ParamSpec
from collections.abc import Callable
import os
import functools
from joblib import Memory

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
            return None
        cacher.save_cache(result, tag=tag)
        cacher.save_config(tag=tag)
        return result

    return wrapper


SELF = TypeVar("SELF", bound=_Cachable)
P = ParamSpec("P")
R = TypeVar("R")


def cached_method(
    cache_tag: str | None = None,
    *,
    verbose: bool = False,
) -> Callable[[Callable[Concatenate[SELF, P], R]], Callable[Concatenate[SELF, P], R]]:
    """
    Cache the results of instance methods.
    Uses joblib's Memory to handle caching with optional custom tag and verbosity.

    In order for the caching to work properly, the method should not depend on any
    instance of the class.

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

    Example
    -------
        >>> class WrongUsage(OperatorMixin):
        ...     def __init__(self):
        ...         self.cacher: _CacherProtocol
        ...         self.x = 1
        ...
        ...     @cached_method()
        ...     def method(self, y: int) -> int:
        ...         self.x += 1
        ...         return y * self.x  # This is dangerous: it dynamically depends on self.x
    """

    def decorator(
        func: Callable[Concatenate[SELF, P], R],
    ) -> Callable[Concatenate[SELF, P], R]:
        @functools.wraps(func)
        def wrapper(self: _Cachable, *args: Any, **kwargs: Any) -> R:
            # Check if caching is allowed based on policy
            policy = self.cacher.policy
            if policy == "OFF":
                # Never use cache, always run the function
                return func(self, *args, **kwargs)

            # For other policies, we'll use the caching mechanism
            cached_methods: dict[str, Callable] = getattr(self, "_cached_methods", {})
            cache_dir = self.cacher.cache_dir
            tag = cache_tag if cache_tag is not None else func.__name__

            if tag not in cached_methods:
                cache_path = os.path.join(cache_dir, tag)
                memory = Memory(cache_path, verbose=verbose)
                # Dev note: __get__ is used to bind the method to this "self" instance.
                # This is necessary because the memory.cache decorator expects a bound method.
                # Otherwise, there is no way to pass the "self" instance during the cache call.
                _func = memory.cache(func.__get__(self, type(self)))
                cached_methods[tag] = _func
                self._cached_methods = cached_methods
            if policy == "MUST":
                if not cached_methods[tag].check_call_in_cache(*args, **kwargs):  # type: ignore
                    raise RuntimeError(
                        f"MUST policy is used for caching, but cache for {tag} does not exist."
                    )
            elif policy == "OVERWRITE":
                cached_methods[tag].clear(warn=False)  # type: ignore

            return cached_methods[tag](*args, **kwargs)  # type: ignore

        return wrapper  # type: ignore

    return decorator
