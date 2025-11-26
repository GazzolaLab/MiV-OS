from __future__ import annotations

__doc__ = """
Caching implementation for MIV data loaders.

This module provides FunctionalCacher, which is specifically designed for
data loaders that use function parameters for configuration.
"""
__all__ = [
    "FunctionalCacher",
]

from typing import Any
from collections.abc import Generator

from collections import OrderedDict
import glob
import json
import os
import pickle as pkl

from ..cachable import BaseCacher, when_policy_is


class FunctionalCacher(BaseCacher):
    """
    Cacher implementation for data loaders using function parameter configuration.

    Serializes function parameters (args/kwargs) to JSON for configuration
    comparison and supports single cache file per tag.
    """

    def _compile_parameters_as_dict(self, params: dict | None = None) -> dict:
        """
        Compile function parameters into a dictionary for configuration comparison.

        Parameters
        ----------
        params : dict | None
            Expected to be a tuple of (args_tuple, kwargs_dict) when not None.
            If None, returns empty dict.

        Returns
        -------
        dict
            Dictionary representation of parameters
        """
        config: dict[str, Any] = OrderedDict()
        if params is None:
            return config
        # params is expected to be a tuple of (args, kwargs) based on original implementation
        # Safe to assume params is a tuple, and all elements are hashable
        if isinstance(params, tuple) and len(params) >= 2:
            args = params[0]
            kwargs = params[1]
        else:
            # Fallback: try to handle as dict or single value
            args = (
                params[0] if hasattr(params, "__getitem__") and len(params) > 0 else ()
            )
            kwargs = (
                params[1] if hasattr(params, "__getitem__") and len(params) > 1 else {}
            )
        for idx, arg in enumerate(args):
            config[f"arg_{idx}"] = arg
        if isinstance(kwargs, dict):
            config.update(kwargs)
        return config

    def check_cached(self, tag: str = "data", *args: Any, **kwargs: Any) -> bool:
        """
        Check if the current configuration matches the cached one.

        This method overrides BaseCacher.check_cached() to handle the `params` parameter.
        It calls the base class method for policy handling.

        Parameters
        ----------
        tag : str, default="data"
            Tag for the cache
        *args : Any
            Additional positional arguments (unused, for compatibility with base class)
        **kwargs : Any
            Additional keyword arguments. May contain 'params' key for function parameters.
        """
        # Extract params from kwargs if present
        params = kwargs.pop("params", None)
        # Ensure tag is not in kwargs to avoid duplicate keyword argument
        kwargs.pop("tag", None)
        # Store params for use in _check_config_matches
        self._current_params = params  # type: ignore[attr-defined]
        # Call base class which handles policies and calls _check_config_matches for ON policy
        # Note: tag is explicitly passed to match base class signature
        return super().check_cached(tag, *args, **kwargs)  # type: ignore[misc]

    def _check_config_matches(
        self, tag: str = "data", *args: Any, **kwargs: Any
    ) -> bool:
        """
        Check if the current function parameters match the cached configuration.

        This method is called by BaseCacher.check_cached() for ON policy only.
        """
        params = getattr(self, "_current_params", None)
        flag = False  # Start as False, only True if config matches AND file exists
        if params is not None:
            current_config = self._compile_parameters_as_dict(params)
            cached_config = self._load_configuration_from_cache(tag)

            if cached_config is not None:
                # Json equality
                flag = current_config == cached_config
        # Also check that cache file exists
        flag = flag and os.path.exists(self.cache_filename(0, tag=tag))
        return flag

    @when_policy_is("ON", "MUST", "OVERWRITE")
    def save_config(self, tag: str = "data", *args: Any, **kwargs: Any) -> bool:
        """
        Save the current configuration to cache.

        Parameters
        ----------
        tag : str, default="data"
            Tag for the cache
        *args : Any
            Additional positional arguments (unused, for compatibility with base class)
        **kwargs : Any
            Additional keyword arguments. May contain 'params' key for function parameters.
        """
        # Extract params from kwargs if present
        params = kwargs.pop("params", None)
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

    def load_cached(self, tag: str = "data") -> Generator[Any]:
        paths = glob.glob(self.cache_filename(0, tag=tag))
        if not paths:
            raise FileNotFoundError(f"No cache found for tag '{tag}'")
        path = paths[0]
        with open(path, "rb") as f:
            self.parent.logger.info(f"Loading cache from: {path}")
            yield pkl.load(f)

    @when_policy_is("ON", "MUST", "OVERWRITE")
    def save_cache(self, values: Any, idx: int = 0, tag: str = "data") -> bool:
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_filename(0, tag=tag), "wb") as f:
            pkl.dump(values, f)
        return True
