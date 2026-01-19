import glob
import logging
import os
import pathlib
import pickle as pkl
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from miv.core.datatype.signal import Signal
from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call


@dataclass
class MockOperator(OperatorMixin):
    tag: str = "test_operator"

    def __post_init__(self):
        super().__init__()
        self.execution_count = 0

    def __call__(self):
        self.execution_count += 1
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        timestamps = np.array([0.0, 1.0, 2.0])
        return Signal(data=data, timestamps=timestamps, rate=1000.0)


@pytest.fixture
def mock_operator(tmp_path: pathlib.Path) -> MockOperator:
    """Fixture that provides a configured MockOperator with temporary path."""
    operator = MockOperator()
    operator.analysis_path = str(tmp_path)
    operator.set_save_path(tmp_path)
    return operator


def test_default_policy_is_on(mock_operator: MockOperator) -> None:
    """Test that default policy should be set to "ON" by default."""
    assert mock_operator.cacher.policy == "ON"


def test_on_policy_executes_and_caches(mock_operator: MockOperator) -> None:
    """
    Test ON policy: __call__ method will be executed with caching.
    At first run, function is executed. At second run, check that the function
    is not executed, and value is retrieved from cache.
    """
    mock_operator.cacher.policy = "ON"

    assert mock_operator.execution_count == 0
    result1 = mock_operator.output()
    assert mock_operator.execution_count == 1
    assert isinstance(result1, Signal)
    assert result1.data.shape == (2, 3)

    result2 = mock_operator.output()
    assert mock_operator.execution_count == 1
    assert isinstance(result2, Signal)
    np.testing.assert_array_equal(result1.data, result2.data)
    np.testing.assert_array_equal(result1.timestamps, result2.timestamps)


def test_off_policy_executes_without_caching(mock_operator: MockOperator) -> None:
    """Test OFF policy: __call__ method will be executed without caching every run."""
    mock_operator.cacher.policy = "OFF"

    assert mock_operator.execution_count == 0
    result1 = mock_operator.output()
    assert mock_operator.execution_count == 1
    assert isinstance(result1, Signal)

    result2 = mock_operator.output()
    assert mock_operator.execution_count == 2
    assert isinstance(result2, Signal)


def test_off_policy_does_not_save_or_alter_cache(mock_operator: MockOperator) -> None:
    """Test OFF policy: __call__ method should not save any cache, or alter existing."""
    mock_operator.cacher.policy = "ON"
    mock_operator.output()
    assert mock_operator.execution_count == 1

    config_file = mock_operator.cacher.config_filename(tag="data")
    cache_file_pattern = mock_operator.cacher.cache_filename(0, tag="data")
    assert os.path.exists(config_file)
    assert os.path.exists(cache_file_pattern)

    config_mtime_before = os.path.getmtime(config_file)
    cache_mtime_before = os.path.getmtime(cache_file_pattern)

    with open(cache_file_pattern, "rb") as f:
        original_cache_content = pkl.load(f)

    mock_operator.cacher.policy = "OFF"
    mock_operator.execution_count = 0
    mock_operator.output()
    assert mock_operator.execution_count == 1

    config_mtime_after = os.path.getmtime(config_file)
    cache_mtime_after = os.path.getmtime(cache_file_pattern)
    assert config_mtime_before == config_mtime_after
    assert cache_mtime_before == cache_mtime_after

    with open(cache_file_pattern, "rb") as f:
        current_cache_content = pkl.load(f)
    np.testing.assert_array_equal(
        original_cache_content.data, current_cache_content.data
    )

    cache_dir = mock_operator.cacher.cache_dir
    cache_files = glob.glob(os.path.join(cache_dir, "cache_data_*.pkl"))
    assert len(cache_files) == 1


def test_must_policy_raises_error_when_cache_not_exists(
    mock_operator: MockOperator,
) -> None:
    """Test MUST policy: __call__ method will raise error if cache does not exist."""
    mock_operator.cacher.policy = "MUST"

    cache_file_pattern = mock_operator.cacher.cache_filename(0, tag="data")
    assert not os.path.exists(cache_file_pattern)

    with pytest.raises(
        FileNotFoundError,
        match="MUST policy is used for caching, but cache does not exist",
    ):
        mock_operator.output()


def test_check_cached_logs_when_policy_is_off(
    mock_operator: MockOperator,
) -> None:
    """Test that check_cached() logs cache status even when policy is OFF."""
    mock_operator.cacher.policy = "OFF"

    # Mock the logger to verify it's called
    log_calls = []
    original_info = mock_operator.logger.info

    def mock_info(msg):
        log_calls.append(msg)
        return original_info(msg)

    with patch.object(mock_operator.logger, "info", side_effect=mock_info):
        result = mock_operator.cacher.check_cached()
        assert result is False

        # Verify that logging was called with the expected message
        assert len(log_calls) > 0, "Expected log message to be called"
        assert any(
            "Caching policy" in msg or "OFF" in msg or "No cache" in msg
            for msg in log_calls
        ), f"Expected cache status log for OFF policy, but got: {log_calls}"


def test_must_policy_retrieves_from_cache(mock_operator: MockOperator) -> None:
    """Test MUST policy: run __call__ method, and change the policy to MUST, and it should retrieve the result from cache."""
    mock_operator.cacher.policy = "ON"
    assert mock_operator.execution_count == 0
    result1 = mock_operator.output()
    assert mock_operator.execution_count == 1
    assert isinstance(result1, Signal)

    cache_file_pattern = mock_operator.cacher.cache_filename(0, tag="data")
    assert os.path.exists(cache_file_pattern)

    mock_operator.cacher.policy = "MUST"
    result2 = mock_operator.output()
    assert mock_operator.execution_count == 1
    assert isinstance(result2, Signal)
    np.testing.assert_array_equal(result1.data, result2.data)
    np.testing.assert_array_equal(result1.timestamps, result2.timestamps)


def test_overwrite_policy_overwrites_existing_cache(
    mock_operator: MockOperator,
) -> None:
    """Test OVERWRITE policy: __call__ method will overwrite existing cache."""
    mock_operator.cacher.policy = "ON"
    assert mock_operator.execution_count == 0
    mock_operator.output()
    assert mock_operator.execution_count == 1

    cache_file_pattern = mock_operator.cacher.cache_filename(0, tag="data")
    config_file = mock_operator.cacher.config_filename(tag="data")
    assert os.path.exists(cache_file_pattern)
    assert os.path.exists(config_file)

    cache_mtime_before = os.path.getmtime(cache_file_pattern)
    config_mtime_before = os.path.getmtime(config_file)

    with open(cache_file_pattern, "rb") as f:
        original_cache_content = pkl.load(f)

    mock_operator.cacher.policy = "OVERWRITE"
    result2 = mock_operator.output()
    assert mock_operator.execution_count == 2
    assert isinstance(result2, Signal)

    cache_mtime_after = os.path.getmtime(cache_file_pattern)
    config_mtime_after = os.path.getmtime(config_file)
    assert cache_mtime_after > cache_mtime_before
    assert config_mtime_after > config_mtime_before

    with open(cache_file_pattern, "rb") as f:
        new_cache_content = pkl.load(f)
    np.testing.assert_array_equal(new_cache_content.data, original_cache_content.data)


def test_overwrite_policy_writes_new_cache_if_not_exists(
    mock_operator: MockOperator,
) -> None:
    """Test OVERWRITE policy: __call__ method will write new cache if cache doesn't exist."""
    mock_operator.cacher.policy = "OVERWRITE"

    cache_file_pattern = mock_operator.cacher.cache_filename(0, tag="data")
    config_file = mock_operator.cacher.config_filename(tag="data")
    assert not os.path.exists(cache_file_pattern)
    assert not os.path.exists(config_file)

    assert mock_operator.execution_count == 0
    result = mock_operator.output()
    assert mock_operator.execution_count == 1
    assert isinstance(result, Signal)

    assert os.path.exists(cache_file_pattern)
    assert os.path.exists(config_file)

    with open(cache_file_pattern, "rb") as f:
        cached_content = pkl.load(f)
    np.testing.assert_array_equal(result.data, cached_content.data)
    np.testing.assert_array_equal(result.timestamps, cached_content.timestamps)


@dataclass
class MockOperatorReturningNone(OperatorMixin):
    """Mock operator that returns None from __call__."""

    tag: str = "test_operator_none"

    def __post_init__(self):
        super().__init__()
        self.execution_count = 0

    def __call__(self):
        self.execution_count += 1
        return None


@pytest.fixture
def mock_operator_returning_none(tmp_path: pathlib.Path) -> MockOperatorReturningNone:
    """Fixture that provides a configured MockOperatorReturningNone with temporary path."""
    operator = MockOperatorReturningNone()
    operator.analysis_path = str(tmp_path)
    operator.set_save_path(tmp_path)
    return operator


def test_cache_not_saved_when_call_returns_none(
    mock_operator_returning_none: MockOperatorReturningNone,
) -> None:
    """Test that cache saving does not happen when __call__ returns None."""
    mock_operator_returning_none.cacher.policy = "ON"

    cache_file_pattern = mock_operator_returning_none.cacher.cache_filename(
        0, tag="data"
    )
    config_file = mock_operator_returning_none.cacher.config_filename(tag="data")
    assert not os.path.exists(cache_file_pattern)
    assert not os.path.exists(config_file)

    assert mock_operator_returning_none.execution_count == 0
    result = mock_operator_returning_none.output()
    assert mock_operator_returning_none.execution_count == 1
    assert result is None

    assert not os.path.exists(cache_file_pattern)
    assert not os.path.exists(config_file)


def test_cache_call_decorator_raises_deprecation_warning(
    tmp_path: pathlib.Path,
) -> None:
    """Test backward compatibility: adding @cache_call decorator should raise DeprecationWarning, but does not alter the functionality."""
    with pytest.warns(DeprecationWarning, match="This decorator is deprecated"):

        @dataclass
        class MockOperatorWithCacheCall(OperatorMixin):
            """Mock operator with @cache_call decorator for backward compatibility test."""

            tag: str = "test_operator_cache_call"

            def __post_init__(self):
                super().__init__()
                self.execution_count = 0

            @cache_call
            def __call__(self):
                self.execution_count += 1
                data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                timestamps = np.array([0.0, 1.0, 2.0])
                return Signal(data=data, timestamps=timestamps, rate=1000.0)

    operator = MockOperatorWithCacheCall()
    operator.analysis_path = str(tmp_path)
    operator.set_save_path(tmp_path)
    operator.cacher.policy = "ON"

    cache_file_pattern = operator.cacher.cache_filename(0, tag="data")
    config_file = operator.cacher.config_filename(tag="data")
    assert not os.path.exists(cache_file_pattern)

    assert operator.execution_count == 0
    result = operator.output()
    assert operator.execution_count == 1
    assert isinstance(result, Signal)

    assert os.path.exists(cache_file_pattern)
    assert os.path.exists(config_file)
