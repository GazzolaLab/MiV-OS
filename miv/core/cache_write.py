"""
Shared rules for persisting computed values through a :class:`~miv.core.cachable._CacherProtocol`.

Used by:

- **Operators** — :class:`~miv.core.operator.operator.OperatorMixin` (scalar) and
  :class:`~miv.core.operator_generator.operator.GeneratorOperatorMixin` (streaming:
  per-chunk writes from :meth:`~miv.core.operator_generator.operator.GeneratorOperatorMixin.output`).
  The legacy :func:`~miv.core.operator_generator.wrapper.cache_generator_call` decorator
  is deprecated and only adapts the runner's ``(idx, *inputs)`` call to ``__call__``.
- **Source / data loaders** — when a path writes through
  :class:`~miv.core.source.cachable.FunctionalCacher`, call this with the same
  *chunk_index* rules; pass loader-specific arguments (e.g. ``params`` for
  ``save_config``) via ``**save_config_kwargs`` on the first chunk only.

This module centralizes *when* ``save_cache`` / ``save_config`` run relative to
chunk index so operator and source paths stay aligned.
"""

from __future__ import annotations

__all__ = ["persist_cacher_result"]

from typing import Any

from .cachable import _CacherProtocol


def persist_cacher_result(
    cacher: _CacherProtocol,
    value: Any,
    *,
    chunk_index: int = 0,
    tag: str = "data",
    **save_config_kwargs: Any,
) -> None:
    """
    Persist one computed chunk through *cacher*.

    Parameters
    ----------
    cacher
        The node's cacher (e.g. :class:`~miv.core.operator.cachable.DataclassCacher`
        or :class:`~miv.core.source.cachable.FunctionalCacher`).
    value
        Payload to pickle; if ``None``, nothing is written.
    chunk_index
        File index for chunked caches. Scalar paths use ``0``. Configuration is
        written only when ``chunk_index == 0`` (first chunk of a logical run).
    tag
        Cache file tag (usually ``"data"``).
    **save_config_kwargs
        Passed to ``cacher.save_config`` when ``chunk_index == 0``. Source loaders
        using :class:`~miv.core.source.cachable.FunctionalCacher` typically pass
        ``params=...``; operator cachers ignore extra keys they do not use.
    """
    if value is None:
        return
    cacher.save_cache(value, idx=chunk_index, tag=tag)
    if chunk_index == 0:
        cacher.save_config(tag=tag, **save_config_kwargs)
