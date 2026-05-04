"""Wrapper decorator for generator operators (legacy signature adapter)."""

from __future__ import annotations

import functools
import warnings
from typing import Any, TypeVar
from collections.abc import Callable

from .operator import GeneratorOperatorMixin

C = TypeVar("C", bound="GeneratorOperatorMixin")


def cache_generator_call(func: Callable) -> Callable:
    """
    Adapt the generator runner's ``(idx, *chunk_inputs)`` call to ``__call__(*chunk_inputs)``.

    .. deprecated::
        Per-chunk cache writes and generator plots run from
        :meth:`~miv.core.operator_generator.operator.GeneratorOperatorMixin.output`.
        Keep this decorator only while migrating subclasses; it will be removed in
        a future release.

    A :exc:`DeprecationWarning` is emitted on the **first** call to each wrapped
    method so importing operators stays quiet until work runs.
    """

    @functools.wraps(func)
    def wrapper(
        self: C,
        idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not getattr(wrapper, "_cg_deprecation_warned", False):  # pragma: no cover
            warnings.warn(
                "cache_generator_call is deprecated: GeneratorOperatorMixin.output() "
                "now applies persist_cacher_result and generator/first-iter plot "
                "callbacks per chunk. This decorator only strips idx from __call__ "
                "until subclasses are updated; remove @cache_generator_call when ready.",
                DeprecationWarning,
                stacklevel=2,
            )
            wrapper._cg_deprecation_warned = True  # type: ignore[attr-defined]

        return func(self, *args, **kwargs)

    return wrapper
