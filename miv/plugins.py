import importlib
import pkgutil

_discovered_plugins = [
    name for finder, name, ispkg in pkgutil.iter_modules() if name.startswith("miv_")
]

for _plugin in _discovered_plugins:
    mdl = importlib.import_module(_plugin)

    if "__all__" in mdl.__dict__:
        names = mdl.__dict__["__all__"]
    else:
        names = [x for x in mdl.__dict__ if not x.startswith("_")]

    # now drag them in
    loads = {k: getattr(mdl, k) for k in names}
    globals().update(loads)
