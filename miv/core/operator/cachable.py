from __future__ import annotations

__doc__ = """
"""
__all__ = ["_Cachable", "DataclassCacher"]

from typing import TYPE_CHECKING, Any, Generator, Literal, Protocol, Union

import collections
import dataclasses
import functools
import glob
import itertools
import json
import os
import pathlib
import pickle as pkl

if TYPE_CHECKING:
    from miv.core.datatype import DataTypes

CACHE_POLICY = Literal["AUTO", "ON", "OFF"]


class _CacherProtocol(Protocol):
    @property
    def cache_dir(self) -> str | pathlib.Path:
        ...

    @property
    def config_filename(self) -> str | pathlib.Path:
        ...

    def cache_filename(self) -> str | pathlib.Path:
        ...

    def load_cached(self) -> Generator[Any, None, None]:
        """Load the cached values."""
        ...

    def save_cache(self, values: Any, idx: int) -> bool:
        ...

    def save_config(self) -> None:
        ...

    def check_cached(self) -> bool:
        """Check if the current configuration is the same as the cached one."""
        ...


class _Cachable(Protocol):
    @property
    def analysis_path(self) -> str | pathlib.Path:
        ...

    @property
    def cacher(self) -> _CacherProtocol:
        ...

    def set_caching_policy(self, policy: CACHE_POLICY) -> None:
        ...

    def run(self, cache_dir: str | pathlib.Path) -> None:
        ...


class SkipCache:
    """
    Always run without saving.
    """

    def __init__(self, parent, cache_dir: str | pathlib.Path):
        super().__init__()

    @property
    def config_filename(self) -> str:
        raise NotImplementedError(
            "If you are using SkipCache, you should not be calling this method."
        )

    def cache_filename(self, idx) -> str:
        raise NotImplementedError(
            "If you are using SkipCache, you should not be calling this method."
        )

    def check_cached(self) -> bool:
        return False

    def save_config(self):
        raise NotImplementedError(
            "If you are using SkipCache, you should not be calling this method."
        )

    def load_cached(self):
        raise NotImplementedError(
            "If you are using SkipCache, you should not be calling this method."
        )

    def save_cache(self, values, idx):
        raise NotImplementedError(
            "If you are using SkipCache, you should not be calling this method."
        )


class DataclassCacher:
    def __init__(self, parent):
        super().__init__()
        self.cache_policy: CACHE_POLICY = "AUTO"  # TODO: make this a property
        self.parent = parent
        self.cache_dir = None  # TODO: Public. Make proper setter

    @property
    def config_filename(self) -> str:
        return os.path.join(self.cache_dir, "config.json")

    def cache_filename(self, idx) -> str:
        index = idx if isinstance(idx, str) else f"{idx:04}"
        return os.path.join(self.cache_dir, f"cache_{index}.pkl")

    def check_cached(self) -> bool:
        current_config = self._compile_configuration_as_dict()
        cached_config = self._load_configuration_from_cache()
        if cached_config is None:
            return False
        return current_config == cached_config  # TODO: fix this

    def _load_configuration_from_cache(self) -> dict:
        if os.path.exists(self.config_filename):
            with open(self.config_filename) as f:
                return json.load(f)
        return None

    def _compile_configuration_as_dict(self) -> dict:
        return dataclasses.asdict(self.parent, dict_factory=collections.OrderedDict)

    def save_config(self):
        config = self._compile_configuration_as_dict()
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.config_filename, "w") as f:
            json.dump(config, f, indent=4)

    def load_cached(self) -> Generator[DataTypes, None, None]:
        paths = glob.glob(self.cache_filename("*"))
        for path in paths:
            with open(path, "rb") as f:
                yield pkl.load(f)

    def save_cache(self, values, idx=0) -> bool:
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_filename(idx), "wb") as f:
            pkl.dump(values, f)
        return True
