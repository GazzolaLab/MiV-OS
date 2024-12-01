from __future__ import annotations

__doc__ = """
"""
__all__ = [
    "_CacherProtocol",
    "_Jsonable",
    "_Cachable",
    "SkipCacher",
    "DataclassCacher",
    "FunctionalCacher",
]

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeVar,
    Union,
)
from collections.abc import Callable, Generator

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

from miv.utils.formatter import TColors
from miv.core.operator.policy import _Runnable

if TYPE_CHECKING:
    from miv.core.datatype import DataTypes

# ON: always use cache
# OFF: never use cache functions (always run) (no cache save)
# MUST: must use cache. If no cache exist, raise error
# OVERWRITE: force run and overwrite cache.
CACHE_POLICY = Literal["AUTO", "ON", "OFF", "MUST", "RUN", "OVERWRITE"]


class _CacherProtocol(Protocol):
    policy: CACHE_POLICY

    @property
    def cache_dir(self) -> str | pathlib.Path: ...

    def load_cached(self, tag: str) -> Generator[Any]:
        """Load the cached values."""
        ...

    def save_cache(self, values: Any, idx: int, tag: str) -> bool: ...

    def check_cached(self, tag: str) -> bool:
        """Check if the current configuration is the same as the cached one."""
        ...


class _Jsonable(Protocol):
    def to_json(self) -> dict[str, Any]: ...


class _Cachable(Protocol):
    @property
    def analysis_path(self) -> str | pathlib.Path: ...

    @property
    def cacher(self) -> _CacherProtocol: ...

    def set_caching_policy(self, policy: CACHE_POLICY) -> None: ...

    def run(self, cache_dir: str | pathlib.Path) -> None: ...


class SkipCacher:
    """
    Always run without saving.
    """

    MSG = "If you are using SkipCache, you should not be calling this method."

    def __init__(self, parent=None, cache_dir=None):
        pass

    def check_cached(self, *args, **kwargs) -> bool:
        return False

    def config_filename(self, *args, **kwargs) -> str:
        raise NotImplementedError(self.MSG)

    def cache_filename(self, *args, **kwargs) -> str:
        raise NotImplementedError(self.MSG)

    def save_config(self, *args, **kwargs):
        raise NotImplementedError(self.MSG)

    def load_cached(self, *args, **kwargs):
        raise NotImplementedError(self.MSG)

    def save_cache(self, *args, kwargs):
        raise NotImplementedError(self.MSG)

    @property
    def cache_dir(self) -> str | pathlib.Path:
        raise NotImplementedError(self.MSG)


F = TypeVar("F", bound=Callable[..., Any])


def when_policy_is(*allowed_policy: CACHE_POLICY) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        # @functools.wraps(func) # TODO: fix this
        def wrapper(self, *args, **kwargs):
            if self.policy in allowed_policy:
                return func(self, *args, **kwargs)
            else:
                return False

        return wrapper

    return decorator


def when_initialized(func: F) -> F:  # TODO: refactor
    # @functools.wraps(func) # TODO: fix this
    def wrapper(self, *args, **kwargs):
        if self.cache_dir is None:
            return False
        else:
            return func(self, *args, **kwargs)

    return wrapper


class BaseCacher:
    """
    Base class for cacher.
    """

    def __init__(self, parent: _Runnable):
        super().__init__()
        self.policy: CACHE_POLICY = "AUTO"  # TODO: make this a property
        self.parent = parent
        self.cache_dir = None  # TODO: Public. Make proper setter

    def config_filename(self, tag="data") -> str:
        return os.path.join(self.cache_dir, f"config_{tag}.json")

    def cache_filename(self, idx, tag="data") -> str:
        index = idx if isinstance(idx, str) else f"{idx:04}"
        if getattr(self.parent.runner, "comm", None) is None:
            mpi_tag = f"{0:03d}"
        else:
            mpi_tag = f"{self.parent.runner.comm.Get_rank():03d}"
        return os.path.join(self.cache_dir, f"cache_{tag}_rank{mpi_tag}_{index}.pkl")

    @when_policy_is("ON", "AUTO", "MUST", "OVERWRITE")
    @when_initialized
    def save_cache(self, values, idx=0, tag="data") -> bool:
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_filename(idx, tag), "wb") as f:
            pkl.dump(values, f)
        return True

    def remove_cache(self):
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def _load_configuration_from_cache(self, tag="data") -> dict:
        path = self.config_filename(tag)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def log_cache_status(self, flag):
        msg = f"Caching policy: {self.policy} - "
        if flag:
            msg += "Cache exist"
        else:
            msg += TColors.red + "No cache" + TColors.reset
        self.parent.logger.info(msg)
        self.parent.logger.info(f"Using runner: {self.parent.runner.__class__} type.")


class DataclassCacher(BaseCacher):
    @when_policy_is("ON", "AUTO", "MUST", "OVERWRITE")
    @when_initialized
    def check_cached(self, tag="data", *args, **kwargs) -> bool:
        if self.policy == "MUST":
            flag = True
        elif self.policy == "OVERWRITE":
            flag = False
        else:
            current_config = self._compile_configuration_as_dict()
            cached_config = self._load_configuration_from_cache(tag=tag)
            if cached_config is None:
                flag = False
            else:
                # Json equality
                flag = current_config == cached_config
        self.log_cache_status(flag)
        return flag

    def _compile_configuration_as_dict(self) -> dict:
        config = dataclasses.asdict(self.parent, dict_factory=collections.OrderedDict)
        for key in config.keys():
            if isinstance(config[key], np.ndarray):
                config[key] = config[key].tostring()
            elif hasattr(config[key], "to_json"):
                config[key] = config[key].to_json()
        return config

    @when_policy_is("ON", "AUTO", "MUST", "OVERWRITE")
    @when_initialized
    def save_config(self, tag="data", *args, **kwargs) -> bool:
        config = self._compile_configuration_as_dict()
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            with open(self.config_filename(tag=tag), "w") as f:
                json.dump(config, f, indent=4)
        except (TypeError, OverflowError):
            raise TypeError(
                "Some property of caching objects are not JSON serializable."
            )
        return True

    @when_initialized
    def load_cached(self, tag="data") -> Generator[DataTypes]:
        paths = glob.glob(self.cache_filename("*", tag=tag))
        paths.sort()
        for path in paths:
            with open(path, "rb") as f:
                self.parent.logger.info(f"Loading cache from: {path}")
                yield pkl.load(f)


class FunctionalCacher(BaseCacher):
    def _compile_parameters_as_dict(self, params=None) -> dict:
        # Safe to assume params is a tuple, and all elements are hashable
        config = collections.OrderedDict()
        if params is None:
            return config
        for idx, arg in enumerate(params[0]):
            config[f"arg_{idx}"] = arg
        config.update(params[1])
        return config

    @when_policy_is("ON", "AUTO", "MUST", "OVERWRITE")
    @when_initialized
    def check_cached(self, params=None, tag="data") -> bool:
        if self.policy == "MUST":
            flag = True
        elif self.policy == "OVERWRITE":
            flag = False
        else:
            flag = True
            if params is not None:
                current_config = self._compile_parameters_as_dict(params)
                cached_config = self._load_configuration_from_cache(tag)

                if cached_config is None:
                    flag = False
                else:
                    # Json equality
                    flag = current_config == cached_config
            flag = flag and os.path.exists(self.cache_filename(0, tag=tag))
        self.log_cache_status(flag)
        return flag

    @when_policy_is("ON", "AUTO", "MUST", "OVERWRITE")
    @when_initialized
    def save_config(self, params=None, tag="data"):
        config = self._compile_parameters_as_dict(params)
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            with open(self.config_filename(tag), "w") as f:
                json.dump(config, f, indent=4)
        except (TypeError, OverflowError):
            raise TypeError(
                "Some property of caching objects are not JSON serializable."
            )
        return True

    @when_initialized
    def load_cached(self, tag="data") -> Generator[DataTypes]:
        path = glob.glob(self.cache_filename(0, tag=tag))[0]
        with open(path, "rb") as f:
            self.parent.logger.info(f"Loading cache from: {path}")
            yield pkl.load(f)

    @when_policy_is("ON", "AUTO", "MUST", "OVERWRITE")
    @when_initialized
    def save_cache(self, values, tag="data") -> bool:
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_filename(0, tag=tag), "wb") as f:
            pkl.dump(values, f)
        return True
