"""Callback mixin for generator operators.

This module provides callback functionality for generator-to-generator operators,
including plotting callbacks that execute during generator iterations.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..callback import BaseCallbackMixin


def _prepare_generator_plot_local(
    _self: Any, args: tuple[Any, ...], kw: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    iter_index = args[0]
    output = args[1] if len(args) > 1 else None
    inputs = args[2] if len(args) > 2 else None
    show = kw.get("show", False)
    save_path = kw.get("save_path")
    return (output, inputs), {
        "show": show,
        "save_path": save_path,
        "index": iter_index,
    }


def _prepare_firstiter_plot_local(
    _self: Any, args: tuple[Any, ...], kw: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    output = args[0] if args else None
    inputs = args[1] if len(args) > 1 else None
    show = kw.get("show", False)
    save_path = kw.get("save_path")
    return (output, inputs), {"show": show, "save_path": save_path}


class GeneratorCallbackMixin(BaseCallbackMixin):
    """
    Streaming plot hook groups: ``generator_plot``, ``firstiter_plot``.

    Call :meth:`~miv.core.callback.BaseCallbackMixin._callback` with those names.
    :class:`~miv.core.operator_generator.operator.GeneratorOperatorMixin` only.
    """

    _callback_group_names: ClassVar[tuple[str, ...]] = (
        "generator_plot",
        "firstiter_plot",
    )
    _callback_group_argument_transforms = {
        "generator_plot": _prepare_generator_plot_local,
        "firstiter_plot": _prepare_firstiter_plot_local,
    }
