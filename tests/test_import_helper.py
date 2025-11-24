"""
Test for require_library decorator in miv.import_helper
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from miv.import_helper import (
    _check_mpi_running,
    no_mpi_support,
    require_library,
)


def test_require_library_raises_error_when_library_not_available():
    """
    require_library should raise ImportError when a required library is not available
    at class instantiation.
    """

    @require_library(["nonexistent_library_xyz"])
    class TestClass:
        def __init__(self):
            pass

    with pytest.raises(ImportError, match="nonexistent_library_xyz"):
        TestClass()


def test_require_library_passes_when_library_available():
    """
    require_library should not raise error when all required libraries are available.
    """

    @require_library(["os"])
    class TestClass:
        def __init__(self):
            self.value = 42

    instance = TestClass()
    assert instance.value == 42


def test_require_library_with_multiple_libraries():
    """
    require_library should check all libraries in the list.
    """

    @require_library(["os", "sys"])
    class TestClass:
        def __init__(self):
            pass

    # Should not raise error when all libraries are available
    instance = TestClass()
    assert instance is not None

    # Should raise error when one library is missing
    @require_library(["os", "nonexistent_library_xyz"])
    class TestClass2:
        def __init__(self):
            pass

    with pytest.raises(ImportError, match="nonexistent_library_xyz"):
        TestClass2()


def test_require_library_default_error_message():
    """
    require_library should use the default error message format when libraries are missing.
    """

    @require_library(["nonexistent_library_xyz"])
    class TestClass:
        def __init__(self):
            pass

    with pytest.raises(ImportError) as exc_info:
        TestClass()

    expected_message = (
        "Required libraries not available: nonexistent_library_xyz. "
        "Please install them to use TestClass."
    )
    assert str(exc_info.value) == expected_message


def test_require_library_default_error_message_multiple_libraries():
    """
    require_library should format multiple missing libraries correctly in error message.
    """

    @require_library(["nonexistent_lib1", "nonexistent_lib2"])
    class TestClass:
        def __init__(self):
            pass

    with pytest.raises(ImportError) as exc_info:
        TestClass()

    expected_message = (
        "Required libraries not available: nonexistent_lib1, nonexistent_lib2. "
        "Please install them to use TestClass."
    )
    assert str(exc_info.value) == expected_message


def test_require_library_custom_error_message():
    """
    require_library should allow providing a custom error message.
    """
    custom_message = "Custom error: Please install the required library to proceed."

    @require_library(["nonexistent_library_xyz"], custom_message)
    class TestClass:
        def __init__(self):
            pass

    with pytest.raises(ImportError) as exc_info:
        TestClass()

    assert str(exc_info.value) == custom_message


def test_require_library_works_on_derived_class():
    """
    require_library should work the same way on a class that is derived from
    a class that is decorated with require_library.
    """

    @require_library(["nonexistent_library_xyz"])
    class BaseClass:
        def __init__(self):
            self.base_value = 42

    class DerivedClass(BaseClass):
        def __init__(self):
            super().__init__()
            self.derived_value = 100

    # Derived class should also raise error when library is missing
    with pytest.raises(ImportError, match="nonexistent_library_xyz"):
        DerivedClass()


def test_require_library_works_on_derived_class_with_available_library():
    """
    require_library should allow derived class instantiation when library is available.
    """

    @require_library(["os"])
    class BaseClass:
        def __init__(self):
            self.base_value = 42

    class DerivedClass(BaseClass):
        def __init__(self):
            super().__init__()
            self.derived_value = 100

    # Derived class should work when library is available
    instance = DerivedClass()
    assert instance.base_value == 42
    assert instance.derived_value == 100


def test_no_mpi_support_passes_when_mpi_not_running():
    """
    no_mpi_support should not raise error when MPI is not running.
    """

    @no_mpi_support
    class TestClass:
        def __init__(self):
            self.value = 42

    # Mock MPI not available or size == 1
    with patch("miv.import_helper._check_mpi_running", return_value=False):
        instance = TestClass()
        assert instance.value == 42


def test_no_mpi_support_raises_error_when_mpi_running_with_mpi4py():
    """
    no_mpi_support should raise RuntimeError when MPI is running (detected via mpi4py).
    """

    @no_mpi_support
    class TestClass:
        def __init__(self):
            pass

    # Mock MPI running (size > 1)
    with patch("miv.import_helper._check_mpi_running", return_value=True):
        with pytest.raises(RuntimeError, match="MPI is not supported"):
            TestClass()


def test_no_mpi_support_works_without_mpi4py():
    """
    no_mpi_support should work even when mpi4py is not installed.
    """

    @no_mpi_support
    class TestClass:
        def __init__(self):
            self.value = 42

    # Mock mpi4py not available and no MPI environment variables
    with patch("miv.import_helper._check_mpi_running", return_value=False):
        instance = TestClass()
        assert instance.value == 42


def test_no_mpi_support_detects_mpi_via_environment_variables():
    """
    no_mpi_support should detect MPI via environment variables when mpi4py is not available.
    """

    @no_mpi_support
    class TestClass:
        def __init__(self):
            pass

    # Mock mpi4py not available but MPI environment variable present
    with patch("miv.import_helper._check_mpi_running", return_value=True):
        with pytest.raises(RuntimeError, match="MPI is not supported"):
            TestClass()


def test_no_mpi_support_does_not_call_init_when_mpi_detected():
    """
    no_mpi_support should raise error before instance creation, so __init__ should not be called.
    """
    init_called = False
    new_called = False

    @no_mpi_support
    class TestClass:
        def __new__(cls, *args, **kwargs):
            nonlocal new_called
            new_called = True
            return super().__new__(cls)

        def __init__(self):
            nonlocal init_called
            init_called = True
            self.value = 42

    # Mock MPI running (size > 1)
    with patch("miv.import_helper._check_mpi_running", return_value=True):
        with pytest.raises(RuntimeError, match="MPI is not supported"):
            TestClass()

    # Verify __new__ and __init__ were never called
    assert new_called is False
    assert init_called is False


def test_no_mpi_support_custom_error_message():
    """
    no_mpi_support should allow providing a custom error message.
    """
    custom_message = "Custom error: This class cannot run in MPI environment."

    @no_mpi_support(custom_message)
    class TestClass:
        def __init__(self):
            pass

    # Mock MPI running (size > 1)
    with patch("miv.import_helper._check_mpi_running", return_value=True):
        with pytest.raises(RuntimeError) as exc_info:
            TestClass()

        assert str(exc_info.value) == custom_message


def test_check_mpi_running_returns_false_when_mpi4py_not_available():
    """
    _check_mpi_running should return False when mpi4py is not available
    and no MPI environment variables are set.
    """
    # Remove any MPI environment variables
    env_vars_to_remove = ["OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "I_MPI_HYDRA_HOST_FILE"]
    original_env = {k: os.environ.get(k) for k in env_vars_to_remove}
    for var in env_vars_to_remove:
        os.environ.pop(var, None)

    try:
        # Mock the import inside _check_mpi_running to raise ImportError
        with patch("builtins.__import__") as mock_import:

            def import_side_effect(name, *args, **kwargs):
                if name == "mpi4py":
                    raise ImportError("No module named 'mpi4py'")
                # For other imports, use the real import
                import builtins

                return builtins.__import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect
            result = _check_mpi_running()
            assert result is False
    finally:
        # Restore environment
        for var, value in original_env.items():
            if value is not None:
                os.environ[var] = value


def test_check_mpi_running_returns_false_when_mpi4py_size_equals_one():
    """
    _check_mpi_running should return False when mpi4py exists but MPI size == 1.
    """
    # Create mock MPI objects
    mock_comm = MagicMock()
    mock_comm.Get_size.return_value = 1
    mock_mpi = MagicMock()
    mock_mpi.MPI.COMM_WORLD = mock_comm

    # Mock the "from mpi4py import MPI" statement
    # When doing "from mpi4py import MPI", Python imports mpi4py first, then gets MPI from it
    original_import = __import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mpi4py":
            if fromlist and "MPI" in fromlist:
                # Return a module with MPI attribute
                return mock_mpi
            # Return the mpi4py module itself
            return mock_mpi
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=mock_import):
        result = _check_mpi_running()
        assert result is False
        mock_comm.Get_size.assert_called_once()


def test_check_mpi_running_returns_true_when_mpi4py_size_greater_than_one():
    """
    _check_mpi_running should return True when mpi4py exists and MPI size > 1.
    """
    # Create mock MPI objects
    mock_comm = MagicMock()
    mock_comm.Get_size.return_value = 4
    mock_mpi = MagicMock()
    mock_mpi.MPI.COMM_WORLD = mock_comm

    # Mock the "from mpi4py import MPI" statement
    original_import = __import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mpi4py":
            if fromlist and "MPI" in fromlist:
                # Return a module with MPI attribute
                return mock_mpi
            # Return the mpi4py module itself
            return mock_mpi
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=mock_import):
        result = _check_mpi_running()
        assert result is True
        mock_comm.Get_size.assert_called_once()
