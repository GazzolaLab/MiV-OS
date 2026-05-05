"""Scalar-operator callback groups (``after_run``, ``plot_``)."""

from __future__ import annotations

from typing import Any, ClassVar

from ..callback import BaseCallbackMixin


def _prepare_after_run_local(
    _self: Any, args: tuple[Any, ...], kw: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return args, kw


def _prepare_plot_local(
    self: Any, args: tuple[Any, ...], kw: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    output = args[0] if args else None
    inputs = args[1] if len(args) > 1 else None
    show = kw.get("show", False)
    save_path = kw.get("save_path")
    if save_path is None:
        save_path = self.analysis_path
    if isinstance(inputs, list) and len(inputs) == 1:
        inputs = inputs[0]
    return (output, inputs), {"show": show, "save_path": save_path}


class ScalarCallbackMixin(BaseCallbackMixin):
    """Mixin for scalar operators: ``after_run`` + ``plot`` hook groups."""

    _callback_group_names: ClassVar[tuple[str, ...]] = ("after_run", "plot")
    _callback_group_argument_transforms = {
        "after_run": _prepare_after_run_local,
        "plot": _prepare_plot_local,
    }
