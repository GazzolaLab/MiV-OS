from __future__ import annotations

__doc__ = """
Caching implementation for MIV operators.

This module provides DataclassCacher, which is specifically designed for
operators that use dataclasses for configuration.

Useful wrapper functions for MIV operators.

.. autofunction:: miv.core.source.wrapper.cached_method
.. autofunction:: miv.core.operator_generator.wrapper.cache_generator_call

"""
__all__ = [
    "DataclassCacher",
]

from typing import TYPE_CHECKING, Any
from collections.abc import Generator

from collections import OrderedDict
import dataclasses
import glob
import json
import os
import pickle as pkl

import numpy as np

from ..cachable import BaseCacher, when_policy_is

if TYPE_CHECKING:
    pass


class DataclassCacher(BaseCacher):
    """
    Cacher implementation for operators using dataclass configuration.

    Serializes operator state (dataclass fields) to JSON for configuration
    comparison and supports multiple cache files for chunked data.
    """

    def _check_config_matches(
        self, tag: str = "data", *args: Any, **kwargs: Any
    ) -> bool:
        """
        Check if the current dataclass configuration matches the cached one.

        This method is called by BaseCacher.check_cached() for ON policy only.
        """
        current_config = self._compile_configuration_as_dict()
        cached_config = self._load_configuration_from_cache(tag=tag)
        if cached_config is None:
            return False
        else:
            # Json equality
            return current_config == cached_config

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

    def load_cached(self, tag: str = "data") -> Generator[Any]:
        paths = glob.glob(self.cache_filename("*", tag=tag))
        paths.sort()
        # For MUST policy, verify cache actually exists
        if self.policy == "MUST" and not paths:
            raise FileNotFoundError(
                f"MUST policy is used for caching, but cache does not exist in {self.cache_dir}"
            )
        for path in paths:
            with open(path, "rb") as f:
                self.parent.logger.info(f"Loading cache from: {path}")
                yield pkl.load(f)
