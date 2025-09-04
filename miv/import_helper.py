from typing import no_type_check
import importlib


@no_type_check
def getter_upon_call(
    module_name: str,
    submodule_alias_paths: dict[str, list[str]],
) -> callable:  # pragma: no cover
    def func(name):  # pragma: no cover
        for k, v in submodule_alias_paths.items():
            if name in v:
                mod = importlib.import_module(f"{module_name}.{k}")
                return getattr(mod, name)
        return importlib.import_module(f"{module_name}.{name}")

    return func
