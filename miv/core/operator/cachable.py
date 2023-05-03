from __future__ import annotations

__doc__ = """
"""
__all__ = [
    "_CacherProtocol",
    "_Jsonable",
    "_Cachable",
    "DataclassCacher",
    "FunctionalCacher",
]

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
import shutil

import numpy as np

if TYPE_CHECKING:
    from miv.core.datatype import DataTypes

CACHE_POLICY = Literal["AUTO", "ON", "OFF", "MUST"]


class _CacherProtocol(Protocol):
    @property
    def cache_dir(self) -> str | pathlib.Path:
        ...

    @property
    def cache_tag(self) -> str:
        ...

    @property
    def cache_called(self) -> bool:
        """Return true if last call was cached."""
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


class _Jsonable(Protocol):
    def to_json(self) -> dict[str, Any]:
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


class SkipCache:  # TODO
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


def when_policy_is(*allowed_policy):
    def decorator(func):
        # @functools.wraps(func) # TODO: fix this
        def wrapper(self, *args, **kwargs):
            if self.policy in allowed_policy:
                return func(self, *args, **kwargs)
            else:
                return False

        return wrapper

    return decorator


def when_initialized(func):  # TODO: refactor
    # @functools.wraps(func) # TODO: fix this
    def wrapper(self, *args, **kwargs):
        if self.cache_dir is None:
            return False
        else:
            return func(self, *args, **kwargs)

    return wrapper


class BaseCacher:
    def __init__(self, parent):
        super().__init__()
        self.policy: CACHE_POLICY = "AUTO"  # TODO: make this a property
        self.parent = parent
        self.cache_dir = None  # TODO: Public. Make proper setter
        self.cache_tag = "data"

        self.cache_called = False

    @property
    def config_filename(self) -> str:
        return os.path.join(self.cache_dir, "config.json")

    def cache_filename(self, idx) -> str:
        index = idx if isinstance(idx, str) else f"{idx:04}"
        return os.path.join(self.cache_dir, f"cache_{self.cache_tag}_{index}.pkl")

    @when_policy_is("ON", "AUTO", "MUST")
    @when_initialized
    def save_cache(self, values, idx=0) -> bool:
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_filename(idx), "wb") as f:
            pkl.dump(values, f)
        return True

    def remove_cache(self):
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)


class DataclassCacher(BaseCacher):
    @when_policy_is("ON", "AUTO", "MUST")
    @when_initialized
    def check_cached(self) -> bool:
        self.cache_called = False  # FIXME: Maybe better place to switch then this
        if self.policy == "MUST":
            return True
        current_config = self._compile_configuration_as_dict()
        cached_config = self._load_configuration_from_cache()
        if cached_config is None:
            flag = False
        else:
            flag = current_config == cached_config  # TODO: fix this
        return flag

    def _load_configuration_from_cache(self) -> dict:
        if os.path.exists(self.config_filename):
            with open(self.config_filename) as f:
                return json.load(f)
        return None

    def _compile_configuration_as_dict(self) -> dict:
        config = dataclasses.asdict(self.parent, dict_factory=collections.OrderedDict)
        for key in config.keys():
            if isinstance(config[key], np.ndarray):
                config[key] = config[key].tostring()
            elif hasattr(config[key], "to_json"):
                config[key] = config[key].to_json()
        return config

    @when_policy_is("ON", "AUTO", "MUST")
    @when_initialized
    def save_config(self):
        config = self._compile_configuration_as_dict()
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            with open(self.config_filename, "w") as f:
                json.dump(config, f, indent=4)
        except (TypeError, OverflowError):
            raise TypeError(
                "Some property of caching objects are not JSON serializable."
            )
        return True

    @when_initialized
    def load_cached(self) -> Generator[DataTypes, None, None]:
        self.cache_called = True
        paths = glob.glob(self.cache_filename("*"))
        for path in paths:
            with open(path, "rb") as f:
                yield pkl.load(f)


class FunctionalCacher(BaseCacher):
    @when_policy_is("ON", "AUTO", "MUST")
    @when_initialized
    def check_cached(self) -> bool:
        if self.policy == "MUST":  # TODO: fix this, remove redundancy
            return True
        flag = os.path.exists(self.cache_filename(0))
        return flag

    @when_policy_is("ON", "AUTO", "MUST")
    @when_initialized
    def save_config(self):
        pass

    @when_initialized
    def load_cached(self) -> Generator[DataTypes, None, None]:
        self.cache_called = True
        path = glob.glob(self.cache_filename(0))[0]
        with open(path, "rb") as f:
            yield pkl.load(f)
