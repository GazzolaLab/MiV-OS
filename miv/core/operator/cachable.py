from __future__ import annotations

__doc__ = """
"""
__all__ = [
    "_CacherProtocol",
    "BaseCacher",
    "DataclassCacher",
    "FunctionalCacher",
]

from typing import TYPE_CHECKING, Any, Literal, Protocol
from collections.abc import Callable, Generator

from collections import OrderedDict
import dataclasses
import glob
import json
import os
import pathlib
import pickle as pkl
import shutil

import numpy as np

from miv.utils.formatter import TColors

if TYPE_CHECKING:
    from .protocol import _Cachable, _Node
    from miv.core.datatype import DataTypes

# ON: Always use cache if cache exist. Otherwise, run and save cache.
# OFF: Never use cache functions, always run. No cache save.
# MUST: must use cache. If no cache exist, raise error
# OVERWRITE: Always run and overwrite cache.
CACHE_POLICY = Literal["ON", "OFF", "MUST", "OVERWRITE"]


class _CacherProtocol(Protocol):
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
    def decorator(
        func: Callable[[_CacherProtocol, Any, Any], bool | DataTypes],
    ) -> Callable:
        # @functools.wraps(func) # TODO: fix this
        def wrapper(
            self: _CacherProtocol, *args: Any, **kwargs: Any
        ) -> bool | DataTypes:
            if self.policy in allowed_policy:
                return func(self, *args, **kwargs)
            else:
                return False

        return wrapper

    return decorator


class BaseCacher:
    """
    Base class for cacher.
    """

    def __init__(self, parent: _Node) -> None:
        super().__init__()
        self.policy: CACHE_POLICY = "ON"
        self.parent = parent
        self.cache_dir: str | pathlib.Path = "results"

    def config_filename(self, tag: str = "data") -> str:
        return os.path.join(self.cache_dir, f"config_{tag}.json")

    def cache_filename(self, idx: int | str, tag: str = "data") -> str:
        index = idx if isinstance(idx, str) else f"{idx:04}"
        if getattr(self.parent.runner, "comm", None) is None:
            mpi_tag = f"{0:03d}"
        else:
            mpi_tag = f"{self.parent.runner.get_run_order():03d}"
        return os.path.join(self.cache_dir, f"cache_{tag}_rank{mpi_tag}_{index}.pkl")

    @when_policy_is("ON", "MUST", "OVERWRITE")
    def save_cache(self, values: Any, idx: int = 0, tag: str = "data") -> bool:
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_filename(idx, tag), "wb") as f:
            pkl.dump(values, f)
        return True

    def remove_cache(self) -> None:
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def _load_configuration_from_cache(self, tag: str = "data") -> dict | str | None:
        path = self.config_filename(tag)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)  # type: ignore[no-any-return]
        return None

    def log_cache_status(self, flag: bool) -> None:
        msg = f"Caching policy: {self.policy} - "
        if flag:
            msg += "Cache exist"
        else:
            msg += TColors.red + "No cache" + TColors.reset
        self.parent.logger.info(msg)


class DataclassCacher(BaseCacher):
    @when_policy_is("ON", "MUST", "OVERWRITE")
    def check_cached(self, tag: str = "data", *args: Any, **kwargs: Any) -> bool:
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

    def _compile_configuration_as_dict(self) -> dict[Any, Any]:
        config: OrderedDict = dataclasses.asdict(self.parent, dict_factory=OrderedDict)  # type: ignore
        for key in config.keys():
            if isinstance(config[key], np.ndarray):
                config[key] = config[key].tostring()
            elif hasattr(config[key], "to_json"):
                config[key] = config[key].to_json()
        return config

    @when_policy_is("ON", "MUST", "OVERWRITE")
    def save_config(self, tag: str = "data", *args: Any, **kwargs: Any) -> bool:
        config = self._compile_configuration_as_dict()
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            with open(self.config_filename(tag=tag), "w") as f:
                json.dump(config, f, indent=4)
        except (TypeError, OverflowError) as err:
            raise TypeError(
                "Some property of caching objects are not JSON serializable."
            ) from err
        return True

    def load_cached(self, tag: str = "data") -> Generator[DataTypes]:
        paths = glob.glob(self.cache_filename("*", tag=tag))
        paths.sort()
        for path in paths:
            with open(path, "rb") as f:
                self.parent.logger.info(f"Loading cache from: {path}")
                yield pkl.load(f)


class FunctionalCacher(BaseCacher):
    def _compile_parameters_as_dict(self, params: dict | None = None) -> dict:
        # Safe to assume params is a tuple, and all elements are hashable
        config: dict[str, Any] = OrderedDict()
        if params is None:
            return config
        for idx, arg in enumerate(params[0]):
            config[f"arg_{idx}"] = arg
        config.update(params[1])
        return config

    @when_policy_is("ON", "MUST", "OVERWRITE")
    def check_cached(self, params: dict | None = None, tag: str = "data") -> bool:
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

    @when_policy_is("ON", "MUST", "OVERWRITE")
    def save_config(self, params: dict | None = None, tag: str = "data") -> bool:
        config = self._compile_parameters_as_dict(params)
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            with open(self.config_filename(tag), "w") as f:
                json.dump(config, f, indent=4)
        except (TypeError, OverflowError) as err:
            raise TypeError(
                "Some property of caching objects are not JSON serializable."
            ) from err
        return True

    def load_cached(self, tag: str = "data") -> Generator[DataTypes]:
        path = glob.glob(self.cache_filename(0, tag=tag))[0]
        with open(path, "rb") as f:
            self.parent.logger.info(f"Loading cache from: {path}")
            yield pkl.load(f)

    @when_policy_is("ON", "MUST", "OVERWRITE")
    def save_cache(self, values: Any, idx: int = 0, tag: str = "data") -> bool:
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_filename(0, tag=tag), "wb") as f:
            pkl.dump(values, f)
        return True
