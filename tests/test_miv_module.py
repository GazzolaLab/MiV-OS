"""
Test that the miv module can be imported without mpi4py available.
"""

import sys
import importlib.util

import pytest


def test_miv_module_loads_without_mpi4py():
    """
    The module miv and miv.core should be able to be imported without mpi4py is available.
    Some implementation and modules may depend on mpi4py, but they should not be loaded as
    part of the normal import process.
    """

    msg = (
        "If you see this error, it means mpi4py was loaded as part of loading miv or miv.core,\n"
        "but it should not be loaded as part of the import process."
    )

    import builtins

    module_names = ["mpi4py", "mpi4py.MPI"]

    # Remove from sys.modules to force re-import attempts to fail
    to_remove = [name for name in module_names if name in sys.modules]
    for name in to_remove:
        sys.modules.pop(name, None)

    # Patch import to raise ImportError for mpi4py
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mpi4py" or name.startswith("mpi4py."):
            raise ImportError(msg)
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = fake_import

    import miv
    import miv.core

    # Unpatch import to restore normal importing for other tests
    builtins.__import__ = orig_import
