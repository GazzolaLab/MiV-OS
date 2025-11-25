from __future__ import annotations

__doc__ = """
Common caching infrastructure for MIV operators and data loaders.

This module provides the base caching functionality that is shared between
operators and data loaders. It includes cache policies, protocols, and base
cacher classes.
"""
__all__ = [
    "_CacherProtocol",
    "BaseCacher",
    "CACHE_POLICY",
    "when_policy_is",
]

from typing import TYPE_CHECKING, Any, Literal, Protocol
from collections.abc import Callable, Generator
from abc import ABC, abstractmethod

import json
import os
import pathlib
import pickle as pkl
import shutil

from miv.utils.formatter import TColors

if TYPE_CHECKING:
    from .protocol import _Cachable

# ON: Always use cache if cache exist. Otherwise, run and save cache.
# OFF: Never use cache functions, always run. No cache save.
# MUST: must use cache. If no cache exist, raise error
# OVERWRITE: Always run and overwrite cache.
CACHE_POLICY = Literal["ON", "OFF", "MUST", "OVERWRITE"]


class _CacherProtocol(Protocol):
    """Protocol defining the interface for cache implementations."""

    policy: CACHE_POLICY
    cache_dir: str | pathlib.Path

    def __init__(self, parent: _Cachable) -> None: ...

    def load_cached(self, tag: str = "data") -> Generator[Any]:
        """Load the cached values."""
        ...

    def save_cache(self, values: Any, idx: int = 0, tag: str = "data") -> bool: ...

    def check_cached(self, tag: str = "data", *args: Any, **kwargs: Any) -> bool:
        """Check if the current configuration is the same as the cached one."""
        ...

    def save_config(self, tag: str = "data", *args: Any, **kwargs: Any) -> bool: ...


def when_policy_is(*allowed_policy: CACHE_POLICY) -> Callable:
    """
    Decorator to restrict method execution to specific cache policies.

    If the current policy is not in the allowed policies, the method returns False.
    """

    def decorator(
        func: Callable[[_CacherProtocol, Any, Any], Any],
    ) -> Callable:
        import functools

        @functools.wraps(func)
        def wrapper(self: _CacherProtocol, *args: Any, **kwargs: Any) -> Any:
            if self.policy in allowed_policy:
                return func(self, *args, **kwargs)
            else:
                return False

        return wrapper

    return decorator


class BaseCacher(ABC):
    """
    Abstract base class for cache implementations.

    Provides common functionality for file-based caching including:
    - Cache policy management
    - File path generation
    - Configuration loading/saving
    - Cache status logging
    """

    def __init__(self, parent: Any) -> None:
        """
        Initialize the cacher.

        Parameters
        ----------
        parent : Any
            The parent object that owns this cacher. Must have `runner` and `logger` attributes.
        """
        super().__init__()
        self.policy: CACHE_POLICY = "ON"
        self.parent = parent
        self.cache_dir: str | pathlib.Path = "results"

    def config_filename(self, tag: str = "data") -> str:
        """Generate the configuration filename for a given tag."""
        return os.path.join(self.cache_dir, f"config_{tag}.json")

    def cache_filename(self, idx: int | str, tag: str = "data") -> str:
        """
        Generate the cache filename for a given index and tag.

        Includes MPI rank information if available.
        """
        index = idx if isinstance(idx, str) else f"{idx:04}"
        if getattr(self.parent.runner, "comm", None) is None:
            mpi_tag = f"{0:03d}"
        else:
            mpi_tag = f"{self.parent.runner.get_run_order():03d}"
        return os.path.join(self.cache_dir, f"cache_{tag}_rank{mpi_tag}_{index}.pkl")

    @when_policy_is("ON", "MUST", "OVERWRITE")
    def save_cache(self, values: Any, idx: int = 0, tag: str = "data") -> bool:
        """
        Save values to cache file.

        Parameters
        ----------
        values : Any
            The values to cache (will be pickled)
        idx : int, default=0
            Index for the cache file
        tag : str, default="data"
            Tag for the cache file

        Returns
        -------
        bool
            True if successful
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_filename(idx, tag), "wb") as f:
            pkl.dump(values, f)
        return True

    def remove_cache(self) -> None:
        """Remove the entire cache directory."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def _load_configuration_from_cache(self, tag: str = "data") -> dict | str | None:
        """
        Load configuration from cache file.

        Returns
        -------
        dict | str | None
            The cached configuration, or None if not found
        """
        path = self.config_filename(tag)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)  # type: ignore[no-any-return]
        return None

    def log_cache_status(self, flag: bool) -> None:
        """Log the cache status."""
        msg = f"Caching policy: {self.policy} - "
        if flag:
            msg += "Cache exist"
        else:
            msg += TColors.red + "No cache" + TColors.reset
        self.parent.logger.info(msg)

    @abstractmethod
    def check_cached(self, tag: str = "data", *args: Any, **kwargs: Any) -> bool:
        """
        Check if the current configuration matches the cached one.

        Must be implemented by subclasses.
        """
        ...

    @abstractmethod
    def save_config(self, tag: str = "data", *args: Any, **kwargs: Any) -> bool:
        """
        Save the current configuration to cache.

        Must be implemented by subclasses.
        """
        ...

    @abstractmethod
    def load_cached(self, tag: str = "data") -> Generator[Any]:
        """
        Load cached values.

        Must be implemented by subclasses.
        """
        ...
