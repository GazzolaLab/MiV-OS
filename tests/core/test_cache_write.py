"""Tests for :mod:`miv.core.cache_write` (shared persist rules for cachers)."""

from __future__ import annotations

from typing import Any

from miv.core.cachable import CACHE_POLICY
from miv.core.cache_write import persist_cacher_result


class RecordingCacher:
    """Minimal cacher that records save_cache / save_config invocations."""

    def __init__(self) -> None:
        self.policy: CACHE_POLICY = "ON"
        self.cache_dir: str = "."
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def save_cache(self, value: Any, idx: int = 0, tag: str = "data") -> bool:
        self.calls.append(("save_cache", (value, idx, tag), {}))
        return True

    def save_config(
        self, tag: str = "data", *args: Any, **kwargs: Any
    ) -> bool:
        self.calls.append(("save_config", (tag,), dict(kwargs)))
        return True


def test_persist_first_chunk_saves_cache_and_config() -> None:
    """Same rules as scalar ``output()``: index 0 writes data and config."""
    c = RecordingCacher()
    persist_cacher_result(c, 42, chunk_index=0, tag="data")
    assert c.calls == [
        ("save_cache", (42, 0, "data"), {}),
        ("save_config", ("data",), {}),
    ]


def test_persist_later_chunk_saves_cache_only() -> None:
    """Generator paths: only chunk 0 triggers ``save_config`` (config once)."""
    c = RecordingCacher()
    persist_cacher_result(c, "b", chunk_index=2, tag="data")
    assert c.calls == [("save_cache", ("b", 2, "data"), {})]


def test_persist_none_is_noop() -> None:
    c = RecordingCacher()
    persist_cacher_result(c, None)
    assert c.calls == []


def test_persist_forwards_save_config_kwargs_on_first_chunk_only() -> None:
    """Source-style :class:`~miv.core.source.cachable.FunctionalCacher` uses ``params`` for config."""
    c = RecordingCacher()
    params = ((), {"x": 1})
    persist_cacher_result(
        c, "payload", chunk_index=0, tag="data", params=params
    )
    assert c.calls == [
        ("save_cache", ("payload", 0, "data"), {}),
        ("save_config", ("data",), {"params": params}),
    ]

    c2 = RecordingCacher()
    persist_cacher_result(
        c2, "chunk", chunk_index=3, tag="data", params=params
    )
    assert c2.calls == [("save_cache", ("chunk", 3, "data"), {})]
