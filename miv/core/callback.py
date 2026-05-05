"""Shared callback utilities and :class:`BaseCallbackMixin` for pipeline nodes.

Scalar hooks (``after_run``, ``plot_``) live in :mod:`miv.core.operator.callback`
(:class:`ScalarCallbackMixin`). Streaming plot hooks live in
:mod:`miv.core.operator_generator.callback` (:class:`GeneratorCallbackMixin`).
"""

from __future__ import annotations

__all__ = ["BaseCallbackMixin"]

from typing import Any, ClassVar
from collections.abc import Callable
from typing_extensions import Self

import types
import inspect
import os
import pathlib

import matplotlib.pyplot as plt

from .loggable import DefaultLoggerMixin
from .cachable import _CacherProtocol, CACHE_POLICY


def execute_callback(
    logger: Any,
    callback: Callable,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Execute one callback and swallow/log any exception."""
    try:
        callback(*args, **kwargs)
    except Exception:
        logger.exception(
            "There was an issue running the following callback. (No exception will be raised during callback.)"
        )


def get_methods_from_feature_classes_by_startswith_str(
    cls: Any, method_name: str
) -> list[Callable]:
    """Return callable attributes on ``cls`` whose names start with ``method_name``."""
    methods = [
        getattr(cls, k)
        for k in dir(cls)
        if k.startswith(method_name) and callable(getattr(cls, k))
    ]
    return methods


def get_methods_from_feature_classes_by_endswith_str(
    cls: Any, method_name: str
) -> list[Callable]:
    """Return callable attributes on ``cls`` whose names end with ``method_name``."""
    methods = [
        getattr(cls, k)
        for k in dir(cls)
        if k.endswith(method_name) and callable(getattr(cls, k))
    ]
    return methods


def invoke_prefixed_callbacks(
    target: Any,
    logger: Any,
    name_prefix: str,
    *call_args: Any,
    **call_kwargs: Any,
) -> None:
    """Invoke all ``target`` callbacks that match ``name_prefix``."""
    for fn in get_methods_from_feature_classes_by_startswith_str(target, name_prefix):
        execute_callback(logger, fn, *call_args, **call_kwargs)


class BaseCallbackMixin(DefaultLoggerMixin):
    """Cacher, paths, ``<<`` attachment, and generic :meth:`_callback` dispatch.

    Subclasses set :attr:`_callback_group_names` and
    :attr:`_callback_group_argument_transforms` for the groups they support.

    Extension note:
    This callback system is intentionally strict and currently optimized for two
    extension families (scalar and generator callbacks), which keeps the design
    easy to reason about for this codebase. When adding new callback groups,
    update both the group-name and argument-transform mappings together.
    Future refactors may introduce a more ergonomic extension surface if the
    number of callback families grows.
    """

    _callback_group_names: ClassVar[tuple[str, ...]] = ()
    _callback_group_argument_transforms: ClassVar[dict[str, Callable]] = {}

    def __init__(
        self,
        *args: Any,
        cacher: _CacherProtocol,
        cache_path: str = ".cache",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.__cache_directory_name: str = cache_path
        self._cacher: _CacherProtocol = cacher

        assert self.tag != "", (
            "All operator must have self.tag attribute for identification."
        )

        self._callback_done = dict.fromkeys(self._callback_group_names, False)

        self.tag: str

    @property
    def cacher(self) -> _CacherProtocol:
        return self._cacher

    @cacher.setter
    def cacher(self, value: _CacherProtocol) -> None:
        policy = self._cacher.policy
        cache_dir = self._cacher.cache_dir
        self._cacher = value
        self._cacher.policy = policy
        self._cacher.cache_dir = cache_dir

    def set_caching_policy(self, policy: CACHE_POLICY) -> None:
        """Set cache read/write policy on the underlying cacher."""
        self.cacher.policy = policy

    def set_callback_done(self, name: str, done: bool) -> None:
        """Set callback done-flag; unknown groups are ignored."""
        if name not in self._callback_done:
            return
        self._callback_done[name] = done

    def __lshift__(self, right: Callable) -> Self:
        """Attach a callback function to this instance (bound when ``self`` arg exists)."""
        if inspect.getfullargspec(right)[0][0] == "self":
            setattr(self, right.__name__, types.MethodType(right, self))
        else:
            setattr(self, right.__name__, right)
        return self

    def set_save_path(
        self,
        path: str | pathlib.Path,
        cache_path: str | pathlib.Path | None = None,
    ) -> None:
        """Configure analysis/cache directories for this node instance."""
        if cache_path is None:
            cache_path = path

        self.analysis_path = os.path.join(path, self.tag.replace(" ", "_"))
        _cache_path = os.path.join(
            cache_path, self.tag.replace(" ", "_"), self.__cache_directory_name
        )
        self.cacher.cache_dir = _cache_path

        os.makedirs(self.analysis_path, exist_ok=True)

    def _callback(
        self, callback_name: str, *args: Any, set_done: bool = True, **kwargs: Any
    ) -> None:
        """Dispatch one callback group by prefix using the configured argument transform."""
        if self._callback_done.get(callback_name, False):
            return
        prefix = f"{callback_name}_"
        prepare_fn = self._callback_group_argument_transforms[callback_name]

        call_args, call_kwargs = prepare_fn(self, args, kwargs)
        invoke_prefixed_callbacks(self, self.logger, prefix, *call_args, **call_kwargs)

        # TODO: Not sure if this is still needed
        if kwargs.get("show", False):
            plt.show()
        plt.close("all")

        self._callback_done[callback_name] = set_done
