from typing import Any, Callable, Type


def append_method(cls: Type[Any], func: Callable[..., Any]) -> None:
    def method(self: object, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    method.__name__ = func.__name__
    setattr(cls, func.__name__, method)
