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


def require_library(libraries: list[str], error_message: str | None = None):
    """
    Class decorator that raises ImportError if required libraries are not available
    when the class is instantiated.

    Parameters
    ----------
    libraries : list[str]
        List of library names to check for availability
    error_message : str | None, optional
        Custom error message to use when libraries are missing.
        If None, uses default message format.

    Returns
    -------
    class decorator
        Decorator that wraps class __init__ to check for library availability

    Examples
    --------
    >>> @require_library(["mpi4py"])
    ... class MPIClass:
    ...     def __init__(self):
    ...         pass

    >>> @require_library(["mpi4py"], "Custom error message")
    ... class MPIClass:
    ...     def __init__(self):
    ...         pass
    """

    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            missing_libraries = []
            for lib in libraries:
                try:
                    importlib.import_module(lib)
                except ImportError:
                    missing_libraries.append(lib)

            if missing_libraries:
                if error_message is not None:
                    raise ImportError(error_message)
                else:
                    missing_str = ", ".join(missing_libraries)
                    raise ImportError(
                        f"Required libraries not available: {missing_str}. "
                        f"Please install them to use {cls.__name__}."
                    )

            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls

    return decorator


def _check_mpi_running() -> bool:
    """
    Check if MPI is currently running.

    Returns
    -------
    bool
        True if MPI is running (size > 1), False otherwise.
    """
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_size() > 1
    except ImportError:
        return False


def no_mpi_support(cls=None, error_message: str | None = None):
    """
    Class decorator that raises RuntimeError if MPI is detected when the class is instantiated.

    This decorator works even when mpi4py is not installed. The error is raised before instance
    creation, preventing both __new__ and __init__ from being called.

    Parameters
    ----------
    cls : type | str | None
        The class to decorate (when used as @no_mpi_support),
        or error message string (when used as @no_mpi_support("message"))
    error_message : str | None, optional
        Custom error message to use when MPI is detected.
        If None, uses default message format.

    Returns
    -------
    type or callable
        The decorated class, or a decorator function if error_message is provided

    Examples
    --------
    >>> @no_mpi_support
    ... class NonMPIClass:
    ...     def __init__(self):
    ...         pass

    >>> @no_mpi_support("Custom error message")
    ... class NonMPIClass:
    ...     def __init__(self):
    ...         pass
    """
    # Handle @no_mpi_support("message") case - cls is the error message string
    if isinstance(cls, str):
        error_message = cls
        cls = None

    def decorator(cls):
        original_new = cls.__new__

        def new_new(cls_instance, *args, **kwargs):
            if _check_mpi_running():
                if error_message is not None:
                    raise RuntimeError(error_message)
                else:
                    raise RuntimeError(
                        f"MPI is not supported for {cls.__name__}. "
                        "This class cannot be used in an MPI environment."
                    )

            # If class has a custom __new__, call it; otherwise use object.__new__
            if original_new is not object.__new__:
                return original_new(cls_instance, *args, **kwargs)
            else:
                return object.__new__(cls_instance)

        cls.__new__ = staticmethod(new_new)
        return cls

    # If cls is None, return decorator (for @no_mpi_support("message"))
    # Otherwise, cls is the class (for @no_mpi_support)
    if cls is None:
        return decorator
    else:
        return decorator(cls)
