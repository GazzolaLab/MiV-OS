"""
Tests for callback functionality in operators.
"""

import pytest
from unittest.mock import Mock, MagicMock

from miv.core.operator.callback import execute_callback


def test_execute_callback_executes_successfully():
    """
    execute_callback should execute the function successfully when there's no error.
    """
    logger = Mock()
    callback = Mock(return_value=42)

    result = execute_callback(logger, callback, "arg1", "arg2", kwarg1="value1")

    callback.assert_called_once_with("arg1", "arg2", kwarg1="value1")
    assert result is None
    logger.warning.assert_not_called()


def test_execute_callback_handles_exceptions_gracefully():
    """
    execute_callback should catch exceptions and not raise them.
    """
    logger = Mock()
    error_message = "Test error"

    def failing_callback():
        raise ValueError(error_message)

    # Should not raise an exception
    result = execute_callback(logger, failing_callback)

    assert result is None
    logger.exception.assert_called_once()


def test_execute_callback_logs_error_message():
    """
    execute_callback should log the error message when an exception occurs.
    """
    logger = Mock()
    error_message = "Callback failed"

    def failing_callback():
        raise RuntimeError(error_message)

    result = execute_callback(logger, failing_callback)

    assert result is None
    logger.exception.assert_called_once()
    # Check that the error message is included in the log call
    call_args = logger.exception.call_args
    assert "callback" in call_args[0][0].lower() or "issue" in call_args[0][0].lower()


def test_execute_callback_returns_none_on_failure():
    """
    execute_callback should return None for any failed cases.
    """
    logger = Mock()

    def failing_callback():
        raise Exception("Any exception")

    result = execute_callback(logger, failing_callback)

    assert result is None


def test_execute_callback_does_not_stop_execution():
    """
    execute_callback should not stop execution even when errors occur.
    """
    logger = Mock()
    execution_count = 0

    def failing_callback():
        nonlocal execution_count
        execution_count += 1
        raise ValueError("Error")

    # Should be able to call multiple times without stopping
    result1 = execute_callback(logger, failing_callback)
    result2 = execute_callback(logger, failing_callback)
    result3 = execute_callback(logger, failing_callback)

    assert result1 is None
    assert result2 is None
    assert result3 is None
    assert execution_count == 3
    assert logger.exception.call_count == 3
