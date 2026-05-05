from typing import Any
from collections.abc import Callable


def append_method(cls: type[Any], func: Callable[..., Any]) -> None:
    def method(self: object, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    method.__name__ = func.__name__
    setattr(cls, func.__name__, method)
