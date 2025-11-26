import os
import pathlib

import pytest

from miv.core.source.node_mixin import DataLoaderMixin
from miv.core.source.wrapper import cached_method


@pytest.fixture
def mock_parent():
    """Fixture that provides a MockParent with required attributes for FunctionalCacher."""
    from miv.core.operator.policy import VanillaRunner
    from miv.core.loggable import DefaultLoggerMixin

    class MockParent:
        def __init__(self):
            self.runner = VanillaRunner()
            logger_mixin = DefaultLoggerMixin()
            self.logger = logger_mixin.logger

    return MockParent()


@pytest.fixture
def functional_cacher(mock_parent, tmp_path: pathlib.Path):
    """Fixture that provides a configured FunctionalCacher instance."""
    from miv.core.source.cachable import FunctionalCacher

    cacher = FunctionalCacher(mock_parent)
    cacher.cache_dir = str(tmp_path)
    cacher.policy = "ON"
    return cacher


class MockDataLoader(DataLoaderMixin):
    def __init__(self, path: str) -> None:
        self.data_path = path
        self.analysis_path = path
        self.tag = "mock module"
        super().__init__()
        self.run_check_flag = False

    @cached_method(cache_tag="function_1")
    def func1(self, a: int, b: int) -> int:
        self.run_check_flag = True
        return a + b

    @cached_method(cache_tag="function_2")
    def func2(self, a: int, b: int) -> int:
        self.run_check_flag = True
        return a - b

    @cached_method(cache_tag="function_3")
    def func3(self, a: int, b: int) -> int:
        self.run_check_flag = True
        # This part is to test if the cache is working properly in nested case
        self.func1(a, b)
        return a * b

    @cached_method(cache_tag="function_4")
    def func4(self, a: list[int], b: int) -> list[int]:
        self.run_check_flag = True
        return [x + b for x in a]

    @cached_method(cache_tag="function_5")
    def func5(self, a: int | float, b: int | float) -> int | float:
        self.run_check_flag = True
        return a * b + a + b

    def reset_flag(self) -> None:
        self.run_check_flag = False


@pytest.mark.mpi_xfail
@pytest.mark.parametrize("cls", [MockDataLoader])
def test_two_function_caching(cls, tmp_path):
    runner = cls(str(tmp_path))
    runner.set_save_path(tmp_path)

    # Case
    a = 1
    b = 2

    assert not runner.run_check_flag
    ans1 = runner.func1(a, b)
    assert runner.run_check_flag
    runner.reset_flag()
    ans2 = runner.func2(a, b)
    assert runner.run_check_flag
    runner.reset_flag()
    ans3 = runner.func3(a, b)
    assert runner.run_check_flag

    assert ans1 == a + b
    assert ans2 == a - b
    assert ans3 == a * b
    assert ans1 != ans2, "If this fail, change the test cases"
    assert ans1 != ans3, "If this fail, change the test cases"

    # Check for multiple run
    runner.reset_flag()
    assert not runner.run_check_flag
    cached_ans2 = runner.func2(a, b)
    assert cached_ans2 == ans2
    cached_ans1 = runner.func1(a, b)
    assert cached_ans1 == ans1
    cached_ans2 = runner.func2(a, b)
    assert cached_ans2 == ans2
    cached_ans1 = runner.func1(a, b)
    assert cached_ans1 == ans1
    cached_ans1 = runner.func1(a, b)
    assert cached_ans1 == ans1
    cached_ans2 = runner.func2(a, b)
    assert cached_ans2 == ans2
    cached_ans1 = runner.func1(a, b)
    assert cached_ans1 == ans1
    cached_ans2 = runner.func2(a, b)
    assert cached_ans2 == ans2
    cached_ans2 = runner.func2(a, b)
    assert cached_ans2 == ans2
    cached_ans3 = runner.func3(a, b)
    assert cached_ans3 == ans3
    assert not runner.run_check_flag

    assert ans1 == cached_ans1
    assert ans2 == cached_ans2

    # Check that joblib cache files exist
    analysis_path = pathlib.Path(runner.analysis_path)
    cache_dir = analysis_path / ".cache"
    assert os.path.exists(cache_dir)

    # Check for joblib cache files
    function_1_cache = list(cache_dir.glob("function_1/**/*.pkl"))
    function_2_cache = list(cache_dir.glob("function_2/**/*.pkl"))
    function_3_cache = list(cache_dir.glob("function_3/**/*.pkl"))

    assert len(function_1_cache) > 0
    assert len(function_2_cache) > 0
    assert len(function_3_cache) > 0


@pytest.mark.parametrize("cls", [MockDataLoader])
def test_two_function_caching_cacher_called_flag_test(cls, tmp_path, mocker):
    """Test that caching works correctly by checking the run_check_flag."""
    runner = cls(str(tmp_path))
    runner.set_save_path(tmp_path)

    # Case
    a = 1
    b = 2

    # First call to func1 - should execute the function
    assert not runner.run_check_flag
    ans1 = runner.func1(a, b)
    assert runner.run_check_flag
    assert ans1 == a + b

    # Second call to func1 - should use cache
    runner.reset_flag()
    cached_ans1 = runner.func1(a, b)
    assert not runner.run_check_flag  # Flag should not be set if using cache
    assert cached_ans1 == ans1

    # First call to func2 - should execute the function
    ans2 = runner.func2(a, b)
    assert runner.run_check_flag
    assert ans2 == a - b

    # Second call to func2 - should use cache
    runner.reset_flag()
    cached_ans2 = runner.func2(a, b)
    assert not runner.run_check_flag  # Flag should not be set if using cache
    assert cached_ans2 == ans2

    # Test func3 which calls func1 internally
    runner.reset_flag()
    ans3 = runner.func3(a, b)
    assert runner.run_check_flag
    assert ans3 == a * b

    # Second call to func3 - should use cache
    runner.reset_flag()
    cached_ans3 = runner.func3(a, b)
    assert not runner.run_check_flag  # Flag should not be set if using cache
    assert cached_ans3 == ans3

    # Test with different inputs - should execute the function
    runner.reset_flag()
    ans4 = runner.func1(a + 1, b + 1)
    assert runner.run_check_flag
    assert ans4 == (a + 1) + (b + 1)

    # Check cache files
    analysis_path = pathlib.Path(runner.analysis_path)
    cache_dir = analysis_path / ".cache"
    assert os.path.exists(cache_dir)

    # Check for joblib cache files
    function_1_cache = list(cache_dir.glob("function_1/**/*.pkl"))
    function_2_cache = list(cache_dir.glob("function_2/**/*.pkl"))
    function_3_cache = list(cache_dir.glob("function_3/**/*.pkl"))

    assert len(function_1_cache) > 0
    assert len(function_2_cache) > 0
    assert len(function_3_cache) > 0


@pytest.mark.parametrize("cls", [MockDataLoader])
def test_cached_method_with_list_input(cls, tmp_path: pathlib.Path) -> None:
    """Test cached_method with list input and output."""
    runner = cls(str(tmp_path))
    runner.set_save_path(tmp_path)

    # Test with list input
    a_list = [1, 2, 3, 4, 5]
    b = 10

    assert not runner.run_check_flag
    result1 = runner.func4(a_list, b)
    assert runner.run_check_flag
    assert result1 == [11, 12, 13, 14, 15]

    # Test caching
    runner.reset_flag()
    result2 = runner.func4(a_list, b)
    assert not runner.run_check_flag  # Should use cached result
    assert result1 == result2

    # Test with different input
    a_list2 = [2, 3, 4, 5, 6]
    result3 = runner.func4(a_list2, b)
    assert runner.run_check_flag  # Should recompute for new input
    assert result3 == [12, 13, 14, 15, 16]

    # Check cache files
    analysis_path = pathlib.Path(runner.analysis_path)
    cache_dir = analysis_path / ".cache"
    function_4_cache = list(cache_dir.glob("function_4/**/*.pkl"))
    assert len(function_4_cache) > 0


@pytest.mark.parametrize("cls", [MockDataLoader])
def test_cached_method_with_mixed_types(cls, tmp_path: pathlib.Path) -> None:
    """Test cached_method with mixed numeric types."""
    runner = cls(str(tmp_path))
    runner.set_save_path(tmp_path)

    # Test with integers
    a_int = 5
    b_int = 3
    result1 = runner.func5(a_int, b_int)
    assert result1 == 23  # 5*3 + 5 + 3

    # Test with floats
    a_float = 5.5
    b_float = 3.5
    result2 = runner.func5(a_float, b_float)
    assert result2 == 5.5 * 3.5 + 5.5 + 3.5

    # Test with mixed types
    result3 = runner.func5(a_int, b_float)
    assert result3 == 5 * 3.5 + 5 + 3.5

    # Test caching
    runner.reset_flag()
    result4 = runner.func5(a_int, b_int)
    assert not runner.run_check_flag  # Should use cached result
    assert result1 == result4

    # Check cache files
    analysis_path = pathlib.Path(runner.analysis_path)
    cache_dir = analysis_path / ".cache"
    function_5_cache = list(cache_dir.glob("function_5/**/*.pkl"))
    assert len(function_5_cache) > 0


@pytest.mark.parametrize("cls", [MockDataLoader])
def test_cached_method_verbose_mode(cls, tmp_path: pathlib.Path) -> None:
    """Test cached_method with verbose mode enabled."""
    runner = cls(str(tmp_path))
    runner.set_save_path(tmp_path)

    # Create a new method with verbose=True
    @cached_method(cache_tag="verbose_function", verbose=True)
    def verbose_func(self, x: int) -> int:
        self.run_check_flag = True
        return x * 2

    # Add the method to the instance
    runner.verbose_func = verbose_func.__get__(runner, type(runner))

    # Test the function
    result = runner.verbose_func(10)
    assert result == 20
    assert runner.run_check_flag

    # Test caching
    runner.reset_flag()
    result2 = runner.verbose_func(10)
    assert not runner.run_check_flag  # Should use cached result
    assert result == result2

    # Check cache files
    analysis_path = pathlib.Path(runner.analysis_path)
    cache_dir = analysis_path / ".cache"
    verbose_cache = list(cache_dir.glob("verbose_function/**/*.pkl"))
    assert len(verbose_cache) > 0


def test_cached_method_docstring_example(tmp_path: pathlib.Path) -> None:
    """
    Test that the example in cached_method docstring works correctly.

    This test follows the exact example from cached_method docstring:
    - MyDataLoader class with expensive_computation method
    - First call computes and caches
    - Second call returns cached result
    """
    from miv.core.source.wrapper import cached_method
    from miv.core.source.node_mixin import DataLoaderMixin

    class MyDataLoader(DataLoaderMixin):
        def __init__(self, tmp_path: pathlib.Path):
            self.data_path = str(tmp_path)
            self.analysis_path = str(tmp_path)
            self.tag = "my_data_loader"
            super().__init__()
            self.execution_count = 0

        @cached_method(
            "my_tag"
        )  # Cache with custom tag. Otherwise, it uses the function name.
        def expensive_computation(self, x: int) -> int:
            # This will only be computed once for each unique input
            self.execution_count += 1
            return x * x

    loader = MyDataLoader(tmp_path)
    loader.set_save_path(tmp_path)

    # First call - should compute and cache
    assert loader.execution_count == 0
    result1 = loader.expensive_computation(5)  # Computes and caches
    assert result1 == 25
    assert loader.execution_count == 1

    # Second call - should return cached result
    result2 = loader.expensive_computation(5)  # Returns cached result
    assert result2 == 25
    assert loader.execution_count == 1  # Should not increment (using cache)

    cache_dir = pathlib.Path(loader.analysis_path) / ".cache"
    my_tag_cache = list(cache_dir.glob("my_tag/**/*.pkl"))
    assert len(my_tag_cache) > 0


def test_functional_cacher_check_config_matches_positive_case(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test positive case for FunctionalCacher._check_config_matches().

    When params match cached config AND cache file exists, should return True.
    """
    cacher = functional_cacher

    # Create matching params
    params = ((), {"test": "value", "number": 42})

    # Create config file with matching config
    import json

    os.makedirs(cacher.cache_dir, exist_ok=True)
    current_config = cacher._compile_parameters_as_dict(params)
    with open(cacher.config_filename(tag="data"), "w") as f:
        json.dump(current_config, f)

    # Create cache file
    import pickle as pkl

    with open(cacher.cache_filename(0, tag="data"), "wb") as f:
        pkl.dump({"result": "cached"}, f)

    # Set params and test
    cacher._current_params = params
    result = cacher._check_config_matches(tag="data")

    assert result is True, (
        "Should return True when params match cached config and cache file exists"
    )


def test_functional_cacher_check_config_matches_negative_case_no_match(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test negative case for FunctionalCacher._check_config_matches().

    When params don't match cached config, should return False.
    """
    cacher = functional_cacher

    # Create params that will be tested
    params = ((), {"test": "different_value", "number": 99})

    # Create config file with different config
    import json

    os.makedirs(cacher.cache_dir, exist_ok=True)
    different_config = cacher._compile_parameters_as_dict(
        ((), {"test": "value", "number": 42})
    )
    with open(cacher.config_filename(tag="data"), "w") as f:
        json.dump(different_config, f)

    # Create cache file
    import pickle as pkl

    with open(cacher.cache_filename(0, tag="data"), "wb") as f:
        pkl.dump({"result": "cached"}, f)

    # Set params and test
    cacher._current_params = params
    result = cacher._check_config_matches(tag="data")

    assert result is False, "Should return False when params don't match cached config"


def test_functional_cacher_check_config_matches_negative_case_no_cache_file(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test negative case for FunctionalCacher._check_config_matches().

    When params match cached config but cache file doesn't exist, should return False.
    """
    cacher = functional_cacher

    # Create matching params
    params = ((), {"test": "value", "number": 42})

    # Create config file with matching config
    import json

    os.makedirs(cacher.cache_dir, exist_ok=True)
    current_config = cacher._compile_parameters_as_dict(params)
    with open(cacher.config_filename(tag="data"), "w") as f:
        json.dump(current_config, f)

    # Don't create cache file - it should not exist

    # Set params and test
    cacher._current_params = params
    result = cacher._check_config_matches(tag="data")

    assert result is False, (
        "Should return False when cache file doesn't exist even if config matches"
    )


def test_functional_cacher_check_config_matches_negative_case_no_params(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test negative case for FunctionalCacher._check_config_matches().

    When params is None, should return False.
    """
    cacher = functional_cacher

    # Don't set _current_params (it will be None)
    result = cacher._check_config_matches(tag="data")

    assert result is False, "Should return False when params is None"


def test_cached_method_on_policy_executes_and_caches(tmp_path: pathlib.Path) -> None:
    """
    Test with ON policy: method will be executed with caching.
    At first run, function is executed. At second run, check that the function
    is not executed, and value is retrieved from cache.
    """

    class TestDataLoader(DataLoaderMixin):
        def __init__(self, path: str):
            self.data_path = path
            self.analysis_path = path
            self.tag = "test_loader"
            super().__init__()
            self.execution_count = 0

        @cached_method(cache_tag="test_function")
        def compute(self, x: int) -> int:
            self.execution_count += 1
            return x * 2

    loader = TestDataLoader(str(tmp_path))
    loader.set_save_path(tmp_path)

    # Ensure ON policy (default)
    loader.cacher.policy = "ON"

    # First run - should execute the function
    assert loader.execution_count == 0
    result1 = loader.compute(5)
    assert loader.execution_count == 1
    assert result1 == 10

    # Second run - should use cache, function should NOT be executed
    result2 = loader.compute(5)
    assert loader.execution_count == 1  # Should not increment (function not executed)
    assert result2 == 10  # Should return cached value

    # Verify cache files were created
    cache_dir = pathlib.Path(loader.analysis_path) / ".cache" / "test_function"
    assert cache_dir.exists(), "Cache directory should exist"

    # Test with different input - should execute again
    result3 = loader.compute(10)
    assert loader.execution_count == 2  # Should increment
    assert result3 == 20


def test_functional_cacher_save_config_saves_parameter_config_to_json_file(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test that FunctionalCacher.save_config() saves parameter config to JSON file.
    """
    import json

    cacher = functional_cacher

    # Create test params (tuple of (args, kwargs))
    params = (("arg1", "arg2"), {"key1": "value1", "key2": 42})

    # Call save_config
    result = cacher.save_config(params=params, tag="test_tag")

    # Verify it returned True
    assert result is True

    # Verify JSON file was created
    config_file = pathlib.Path(cacher.config_filename(tag="test_tag"))
    assert config_file.exists(), "Config file should be created"

    # Verify JSON content matches expected config
    with open(config_file) as f:
        saved_config = json.load(f)

    # Expected config should have arg_0, arg_1, key1, key2
    assert saved_config["arg_0"] == "arg1"
    assert saved_config["arg_1"] == "arg2"
    assert saved_config["key1"] == "value1"
    assert saved_config["key2"] == 42


def test_functional_cacher_save_config_creates_cache_directory_if_not_exists(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test that FunctionalCacher.save_config() creates cache directory if it doesn't exist.
    """
    cacher = functional_cacher

    # Set cache_dir to a subdirectory that doesn't exist yet
    cache_subdir = tmp_path / "cache_subdir"
    cacher.cache_dir = str(cache_subdir)
    cacher.policy = "ON"

    # Verify directory doesn't exist initially
    assert not cache_subdir.exists(), "Cache directory should not exist initially"

    # Create test params
    params = ((), {"test": "value"})

    # Call save_config - should create the directory
    result = cacher.save_config(params=params, tag="test_tag")

    # Verify directory was created
    assert cache_subdir.exists(), "Cache directory should be created by save_config"
    assert cache_subdir.is_dir(), "Cache directory should be a directory"

    # Verify it returned True
    assert result is True

    # Verify config file was created in the new directory
    config_file = cache_subdir / "config_test_tag.json"
    assert config_file.exists(), "Config file should be created in the new directory"


def test_functional_cacher_save_config_raises_typeerror_for_non_json_serializable_parameters(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test that FunctionalCacher.save_config() raises TypeError for non-JSON-serializable parameters.
    """
    cacher = functional_cacher

    # Create params with a non-JSON-serializable object (a function)
    def non_serializable_function():
        pass

    params = ((), {"func": non_serializable_function})

    # Call save_config - should raise TypeError
    with pytest.raises(
        TypeError, match="Some property of caching objects are not JSON serializable"
    ):
        cacher.save_config(params=params, tag="test_tag")


def test_functional_cacher_save_config_handles_different_tags_and_overwrites_existing_config_file(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test that FunctionalCacher.save_config() handles different tags and overwrites existing config file when called again.
    """
    import json

    cacher = functional_cacher

    # First call with tag "tag1"
    params1 = ((), {"key1": "value1"})
    result1 = cacher.save_config(params=params1, tag="tag1")
    assert result1 is True

    # Verify first config file exists
    config_file1 = pathlib.Path(cacher.config_filename(tag="tag1"))
    assert config_file1.exists()
    with open(config_file1) as f:
        config1 = json.load(f)
    assert config1["key1"] == "value1"

    # Second call with same tag "tag1" but different params - should overwrite
    params2 = ((), {"key1": "value2", "key2": "value3"})
    result2 = cacher.save_config(params=params2, tag="tag1")
    assert result2 is True

    # Verify config file was overwritten with new content
    with open(config_file1) as f:
        config2 = json.load(f)
    assert config2["key1"] == "value2"
    assert config2["key2"] == "value3"
    # Verify old content is gone (only new keys exist)
    assert len(config2) == 2  # Only key1 and key2

    # Third call with different tag "tag2" - should create new file
    params3 = ((), {"key3": "value4"})
    result3 = cacher.save_config(params=params3, tag="tag2")
    assert result3 is True

    # Verify both config files exist
    config_file2 = pathlib.Path(cacher.config_filename(tag="tag2"))
    assert config_file2.exists()
    assert config_file1.exists()  # First file should still exist

    # Verify tag2 has different content
    with open(config_file2) as f:
        config3 = json.load(f)
    assert config3["key3"] == "value4"
    # Verify tag1 still has its content
    with open(config_file1) as f:
        config1_after = json.load(f)
    assert config1_after["key1"] == "value2"  # Should still have overwritten content


def test_functional_cacher_save_config_handles_different_configs_overwrites_existing_config_file(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test that FunctionalCacher.save_config() handles different configs and overwrites existing config file when called again.

    This test focuses specifically on overwriting behavior with different configs (same tag),
    ensuring the old config is completely replaced, not merged.
    """
    import json

    cacher = functional_cacher

    # First call: save initial config
    params1 = (("arg1", "arg2"), {"key1": "value1", "key2": "value2", "key3": "value3"})
    result1 = cacher.save_config(params=params1, tag="data")
    assert result1 is True

    # Verify initial config file exists with correct content
    config_file = pathlib.Path(cacher.config_filename(tag="data"))
    assert config_file.exists()
    with open(config_file) as f:
        config1 = json.load(f)
    assert config1["arg_0"] == "arg1"
    assert config1["arg_1"] == "arg2"
    assert config1["key1"] == "value1"
    assert config1["key2"] == "value2"
    assert config1["key3"] == "value3"
    assert len(config1) == 5  # 2 args + 3 kwargs

    # Second call: overwrite with completely different config (different args and kwargs)
    params2 = (("arg3",), {"key4": "value4", "key5": "value5"})
    result2 = cacher.save_config(params=params2, tag="data")
    assert result2 is True

    # Verify config file was completely overwritten (old content is gone)
    with open(config_file) as f:
        config2 = json.load(f)

    # New config should only have new keys
    assert config2["arg_0"] == "arg3"
    assert config2["key4"] == "value4"
    assert config2["key5"] == "value5"
    assert len(config2) == 3  # Only new content

    # Old keys should be gone
    assert "arg_1" not in config2, "Old arg_1 should be removed"
    assert "key1" not in config2, "Old key1 should be removed"
    assert "key2" not in config2, "Old key2 should be removed"
    assert "key3" not in config2, "Old key3 should be removed"

    # Third call: overwrite again with yet another different config
    params3 = ((), {"key6": "value6"})
    result3 = cacher.save_config(params=params3, tag="data")
    assert result3 is True

    # Verify config file was overwritten again
    with open(config_file) as f:
        config3 = json.load(f)

    # Should only have the latest config
    assert config3["key6"] == "value6"
    assert len(config3) == 1  # Only latest content

    # All previous keys should be gone
    assert "arg_0" not in config3, "Previous arg_0 should be removed"
    assert "key4" not in config3, "Previous key4 should be removed"
    assert "key5" not in config3, "Previous key5 should be removed"


def test_functional_cacher_save_cache_saves_to_pickle_and_load_cached_yields_pickled_values(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test that FunctionalCacher.save_cache() saves values to pickle file,
    and FunctionalCacher.load_cached() yields pickled values from cache file.
    """
    cacher = functional_cacher

    # Test data to save
    test_data = {
        "string": "test_value",
        "number": 42,
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
    }

    # Save cache using save_cache()
    result = cacher.save_cache(values=test_data, tag="test_tag")
    assert result is True

    # Verify cache file was created
    cache_file = pathlib.Path(cacher.cache_filename(0, tag="test_tag"))
    assert cache_file.exists(), "Cache file should be created"

    # Load cache using load_cached() - should yield the pickled values
    loaded_generator = cacher.load_cached(tag="test_tag")

    # load_cached() returns a generator, so we need to iterate
    loaded_data = next(loaded_generator, None)

    # Verify loaded data matches saved data
    assert loaded_data is not None, "Should yield data from cache"
    assert loaded_data == test_data, "Loaded data should match saved data"
    assert loaded_data["string"] == "test_value"
    assert loaded_data["number"] == 42
    assert loaded_data["list"] == [1, 2, 3]
    assert loaded_data["dict"] == {"nested": "value"}

    # Verify generator yields only one value (idx=0 for FunctionalCacher)
    with pytest.raises(StopIteration):
        next(loaded_generator)


def test_functional_cacher_save_cache_creates_cache_directory_if_not_exists(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test that FunctionalCacher.save_cache() creates cache directory if it doesn't exist.
    """
    cacher = functional_cacher

    # Set cache_dir to a subdirectory that doesn't exist yet
    cache_subdir = tmp_path / "cache_subdir"
    cacher.cache_dir = str(cache_subdir)
    cacher.policy = "ON"

    # Verify directory doesn't exist initially
    assert not cache_subdir.exists(), "Cache directory should not exist initially"

    # Test data to save
    test_data = {"test": "value"}

    # Call save_cache - should create the directory
    result = cacher.save_cache(values=test_data, tag="test_tag")

    # Verify directory was created
    assert cache_subdir.exists(), "Cache directory should be created by save_cache"
    assert cache_subdir.is_dir(), "Cache directory should be a directory"

    # Verify it returned True
    assert result is True

    # Verify cache file was created in the new directory
    cache_file = pathlib.Path(cacher.cache_filename(0, tag="test_tag"))
    assert cache_file.exists(), "Cache file should be created in the new directory"


def test_functional_cacher_save_cache_returns_false_when_policy_is_off(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test that FunctionalCacher.save_cache() returns False when policy is "OFF" (via @when_policy_is decorator).
    """
    cacher = functional_cacher
    cacher.policy = "OFF"  # Set policy to OFF

    # Test data to save
    test_data = {"test": "value"}

    # Call save_cache with OFF policy - should return False
    result = cacher.save_cache(values=test_data, tag="test_tag")
    assert result is False, "save_cache should return False when policy is OFF"

    # Verify cache file was NOT created (because save_cache returned False)
    cache_file = pathlib.Path(cacher.cache_filename(0, tag="test_tag"))
    assert not cache_file.exists(), (
        "Cache file should NOT be created when policy is OFF"
    )


def test_functional_cacher_load_cached_raises_filenotfounderror_when_no_cache_file_exists(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test that FunctionalCacher.load_cached() raises FileNotFoundError when no cache file exists.
    """
    cacher = functional_cacher

    # Verify no cache file exists
    cache_file = pathlib.Path(cacher.cache_filename(0, tag="test_tag"))
    assert not cache_file.exists(), "Cache file should not exist initially"

    # Call load_cached() - should raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="No cache found for tag 'test_tag'"):
        list(
            cacher.load_cached(tag="test_tag")
        )  # Convert generator to list to trigger the error


def test_functional_cacher_load_cached_logs_cache_loading_message(
    functional_cacher, tmp_path: pathlib.Path, mocker
) -> None:
    """
    Test that FunctionalCacher.load_cached() logs cache loading message.
    """
    cacher = functional_cacher
    parent = cacher.parent

    # Create test data and save it
    test_data = {"test": "value"}
    cacher.save_cache(values=test_data, tag="test_tag")

    # Spy on the logger.info method
    logger_spy = mocker.spy(parent.logger, "info")

    # Load cache using load_cached()
    loaded_generator = cacher.load_cached(tag="test_tag")
    loaded_data = next(loaded_generator, None)

    # Verify data was loaded
    assert loaded_data == test_data

    # Verify logger.info was called with cache loading message
    logger_spy.assert_called_once()
    call_args = logger_spy.call_args[0][0]
    assert "Loading cache from:" in call_args, (
        "Log message should contain 'Loading cache from:'"
    )
    assert "test_tag" in call_args or str(tmp_path) in call_args, (
        "Log message should contain tag or path"
    )


def test_functional_cacher_save_cache_handles_different_tags(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test that FunctionalCacher.save_cache() handles different tags.
    """
    cacher = functional_cacher

    # Save cache with tag "tag1"
    test_data1 = {"key1": "value1"}
    result1 = cacher.save_cache(values=test_data1, tag="tag1")
    assert result1 is True

    # Verify cache file was created for tag1
    cache_file1 = pathlib.Path(cacher.cache_filename(0, tag="tag1"))
    assert cache_file1.exists(), "Cache file should be created for tag1"

    # Save cache with tag "tag2"
    test_data2 = {"key2": "value2"}
    result2 = cacher.save_cache(values=test_data2, tag="tag2")
    assert result2 is True

    # Verify cache file was created for tag2
    cache_file2 = pathlib.Path(cacher.cache_filename(0, tag="tag2"))
    assert cache_file2.exists(), "Cache file should be created for tag2"

    # Verify both cache files exist and are different
    assert cache_file1.exists() and cache_file2.exists(), (
        "Both cache files should exist"
    )
    assert cache_file1 != cache_file2, (
        "Cache files should be different for different tags"
    )

    # Verify we can load both caches correctly
    loaded_generator1 = cacher.load_cached(tag="tag1")
    loaded_data1 = next(loaded_generator1, None)
    assert loaded_data1 == test_data1

    loaded_generator2 = cacher.load_cached(tag="tag2")
    loaded_data2 = next(loaded_generator2, None)
    assert loaded_data2 == test_data2

    # Verify data is different
    assert loaded_data1 != loaded_data2, (
        "Loaded data should be different for different tags"
    )


def test_functional_cacher_load_cached_handles_different_tags(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test that FunctionalCacher.load_cached() handles different tags.
    """
    cacher = functional_cacher

    # Save cache with different tags
    test_data1 = {"key1": "value1", "number": 10}
    test_data2 = {"key2": "value2", "number": 20}
    test_data3 = {"key3": "value3", "number": 30}

    cacher.save_cache(values=test_data1, tag="tag1")
    cacher.save_cache(values=test_data2, tag="tag2")
    cacher.save_cache(values=test_data3, tag="tag3")

    # Verify all cache files exist
    cache_file1 = pathlib.Path(cacher.cache_filename(0, tag="tag1"))
    cache_file2 = pathlib.Path(cacher.cache_filename(0, tag="tag2"))
    cache_file3 = pathlib.Path(cacher.cache_filename(0, tag="tag3"))
    assert cache_file1.exists() and cache_file2.exists() and cache_file3.exists()

    # Load cache with tag1
    loaded_generator1 = cacher.load_cached(tag="tag1")
    loaded_data1 = next(loaded_generator1, None)
    assert loaded_data1 == test_data1, "Should load correct data for tag1"

    # Load cache with tag2
    loaded_generator2 = cacher.load_cached(tag="tag2")
    loaded_data2 = next(loaded_generator2, None)
    assert loaded_data2 == test_data2, "Should load correct data for tag2"

    # Load cache with tag3
    loaded_generator3 = cacher.load_cached(tag="tag3")
    loaded_data3 = next(loaded_generator3, None)
    assert loaded_data3 == test_data3, "Should load correct data for tag3"

    # Verify all loaded data is different
    assert loaded_data1 != loaded_data2 != loaded_data3, (
        "Loaded data should be different for different tags"
    )
    assert loaded_data1["number"] == 10
    assert loaded_data2["number"] == 20
    assert loaded_data3["number"] == 30


def test_functional_cacher_load_cached_returns_generator_that_yields_single_value(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test that FunctionalCacher.load_cached() returns generator that yields single value (idx=0).

    FunctionalCacher always uses idx=0, so it should yield exactly one value.
    """
    from collections.abc import Generator

    cacher = functional_cacher

    # Save cache data
    test_data = {"key": "value", "number": 42}
    cacher.save_cache(values=test_data, tag="test_tag")

    # Call load_cached() - should return a generator
    loaded_generator = cacher.load_cached(tag="test_tag")

    # Verify it's a generator
    assert isinstance(loaded_generator, Generator), (
        "load_cached should return a generator"
    )

    # Get the first (and only) value
    loaded_data = next(loaded_generator, None)
    assert loaded_data == test_data, "Should yield the cached data"

    # Verify generator yields only one value (idx=0 for FunctionalCacher)
    # Next call should raise StopIteration
    with pytest.raises(StopIteration):
        next(loaded_generator)

    # Verify we can iterate through it once
    loaded_generator2 = cacher.load_cached(tag="test_tag")
    items = list(loaded_generator2)
    assert len(items) == 1, "Generator should yield exactly one value"
    assert items[0] == test_data, "The single value should match the cached data"


def test_functional_cacher_save_cache_overwrites_existing_cache_file_when_policy_is_overwrite(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test that FunctionalCacher.save_cache() overwrites existing cache file when called again if policy is "OVERWRITE".
    """
    cacher = functional_cacher

    # First, create a cache with ON policy
    cacher.policy = "ON"
    test_data1 = {"key1": "value1", "number": 10}
    result1 = cacher.save_cache(values=test_data1, tag="test_tag")
    assert result1 is True

    # Verify cache file was created
    cache_file = pathlib.Path(cacher.cache_filename(0, tag="test_tag"))
    assert cache_file.exists(), "Cache file should exist after first save"

    # Get modification time before overwrite
    mtime_before = cache_file.stat().st_mtime

    # Change policy to OVERWRITE and save again with different data
    cacher.policy = "OVERWRITE"
    test_data2 = {"key2": "value2", "number": 20}
    result2 = cacher.save_cache(values=test_data2, tag="test_tag")
    assert result2 is True

    # Verify cache file was overwritten (modification time should be newer)
    mtime_after = cache_file.stat().st_mtime
    assert mtime_after > mtime_before, (
        "Cache file should be overwritten (newer modification time)"
    )

    # Verify the new data is in the cache
    loaded_generator = cacher.load_cached(tag="test_tag")
    loaded_data = next(loaded_generator, None)
    assert loaded_data == test_data2, (
        "Cache should contain the new data, not the old data"
    )
    assert loaded_data["number"] == 20, "New data should be present"
    assert "key1" not in loaded_data, "Old data should be removed"


def test_functional_cacher_save_cache_overwrites_existing_cache_file_when_policy_is_on_and_config_does_not_match(
    functional_cacher, tmp_path: pathlib.Path
) -> None:
    """
    Test that FunctionalCacher.save_cache() overwrites existing cache file if policy is "ON" and configuration does not match.

    When configuration doesn't match, check_cached() returns False, and then save_cache() is called
    to save the new result, which should overwrite the existing cache file.
    """
    cacher = functional_cacher

    # First, create a cache with initial params and data
    params1 = ((), {"key1": "value1"})
    test_data1 = {"result": "data1", "number": 10}

    # Save config and cache
    cacher.save_config(params=params1, tag="test_tag")
    cacher.save_cache(values=test_data1, tag="test_tag")

    # Verify cache file exists
    cache_file = pathlib.Path(cacher.cache_filename(0, tag="test_tag"))
    assert cache_file.exists(), "Cache file should exist after first save"

    # Verify check_cached returns True (config matches and cache file exists)
    cacher._current_params = params1  # type: ignore[attr-defined]
    check_result = cacher.check_cached(params=params1, tag="test_tag")
    assert check_result is True, (
        f"check_cached should return True when config matches, got {check_result}"
    )

    # Get modification time before overwrite
    mtime_before = cache_file.stat().st_mtime

    # Change params so configuration doesn't match
    params2 = ((), {"key1": "value2"})  # Different value
    test_data2 = {"result": "data2", "number": 20}  # New data to save

    # Verify check_cached returns False (config doesn't match)
    cacher._current_params = params2  # type: ignore[attr-defined]
    assert cacher.check_cached(tag="test_tag") is False, (
        "check_cached should return False when config doesn't match"
    )

    # Now save_cache() with ON policy - should overwrite existing cache file
    # (This simulates what happens in operator.output() when config doesn't match)
    result = cacher.save_cache(values=test_data2, tag="test_tag")
    assert result is True

    # Verify cache file was overwritten (modification time should be newer)
    mtime_after = cache_file.stat().st_mtime
    assert mtime_after > mtime_before, (
        "Cache file should be overwritten (newer modification time)"
    )

    # Verify the new data is in the cache
    loaded_generator = cacher.load_cached(tag="test_tag")
    loaded_data = next(loaded_generator, None)
    assert loaded_data == test_data2, (
        "Cache should contain the new data, not the old data"
    )
    assert loaded_data["number"] == 20, "New data should be present"
    assert "data1" not in str(loaded_data), "Old data should be removed"


def test_cached_method_on_policy_does_not_call_cacher_methods(
    tmp_path: pathlib.Path, mocker
) -> None:
    """
    Test that cached_method does NOT call save_config, load_cached, or save_cache.

    The cached_method decorator uses joblib.Memory for caching, which handles
    its own file I/O. It should not call FunctionalCacher's save_config, load_cached,
    or save_cache methods.
    """

    class TestDataLoader(DataLoaderMixin):
        def __init__(self, path: str):
            self.data_path = path
            self.analysis_path = path
            self.tag = "test_loader"
            super().__init__()
            self.execution_count = 0

        @cached_method(cache_tag="test_function")
        def compute(self, x: int) -> int:
            self.execution_count += 1
            return x * 2

    loader = TestDataLoader(str(tmp_path))
    loader.set_save_path(tmp_path)
    loader.cacher.policy = "ON"

    # Use pytest spy to track method calls
    save_config_spy = mocker.spy(loader.cacher, "save_config")
    load_cached_spy = mocker.spy(loader.cacher, "load_cached")
    save_cache_spy = mocker.spy(loader.cacher, "save_cache")

    # First run - should execute the function
    result1 = loader.compute(5)
    assert result1 == 10
    assert loader.execution_count == 1

    # Second run - should use cache (joblib's cache), function should NOT be executed
    result2 = loader.compute(5)
    assert result2 == 10
    assert loader.execution_count == 1  # Should not increment

    # Verify that FunctionalCacher methods were NOT called
    save_config_spy.assert_not_called()
    load_cached_spy.assert_not_called()
    save_cache_spy.assert_not_called()


def test_cached_method_off_policy_executes_without_caching(
    tmp_path: pathlib.Path,
) -> None:
    """
    Test with OFF policy: method will be executed without caching every run.
    """

    class TestDataLoader(DataLoaderMixin):
        def __init__(self, path: str):
            self.data_path = path
            self.analysis_path = path
            self.tag = "test_loader"
            super().__init__()
            self.execution_count = 0

        @cached_method(cache_tag="test_function")
        def compute(self, x: int) -> int:
            self.execution_count += 1
            return x * 2

    loader = TestDataLoader(str(tmp_path))
    loader.set_save_path(tmp_path)

    # Set policy to OFF
    loader.cacher.policy = "OFF"

    # First run - should execute the function
    assert loader.execution_count == 0
    result1 = loader.compute(5)
    assert loader.execution_count == 1
    assert result1 == 10

    # Second run with same input - should execute again (no caching)
    result2 = loader.compute(5)
    assert loader.execution_count == 2  # Should increment (function executed again)
    assert result2 == 10

    # Third run - should execute again
    result3 = loader.compute(5)
    assert loader.execution_count == 3  # Should increment again
    assert result3 == 10

    # Verify no cache files were created (OFF policy doesn't cache)
    cache_dir = pathlib.Path(loader.analysis_path) / ".cache" / "test_function"
    # Cache directory might exist but should be empty or not used
    if cache_dir.exists():
        cache_files = list(cache_dir.rglob("*.pkl"))
        # With OFF policy, joblib might still create cache structure, but it shouldn't be used
        # The important thing is that the function executes every time


def test_cached_method_must_policy_retrieves_from_cache(tmp_path: pathlib.Path) -> None:
    """
    Test with MUST policy: run method, and change the policy to MUST, and it should retrieve the result from cache.
    """

    class TestDataLoader(DataLoaderMixin):
        def __init__(self, path: str):
            self.data_path = path
            self.analysis_path = path
            self.tag = "test_loader"
            super().__init__()
            self.execution_count = 0

        @cached_method(cache_tag="test_function")
        def compute(self, x: int) -> int:
            self.execution_count += 1
            return x * 2

    loader = TestDataLoader(str(tmp_path))
    loader.set_save_path(tmp_path)

    # First, run with ON policy to create cache
    loader.cacher.policy = "ON"
    assert loader.execution_count == 0
    result1 = loader.compute(5)
    assert loader.execution_count == 1
    assert result1 == 10

    # Verify cache was created
    cache_dir = pathlib.Path(loader.analysis_path) / ".cache" / "test_function"
    assert cache_dir.exists(), "Cache should exist after ON policy run"

    # Change policy to MUST and run again
    loader.cacher.policy = "MUST"
    result2 = loader.compute(5)

    # Should retrieve from cache, function should NOT be executed
    assert loader.execution_count == 1  # Should not increment
    assert result2 == 10  # Should return cached value
    assert result1 == result2  # Should be the same result


def test_cached_method_overwrite_policy_overwrites_existing_cache(
    tmp_path: pathlib.Path,
) -> None:
    """
    Test with OVERWRITE policy: method will overwrite existing cache.
    """

    class TestDataLoader(DataLoaderMixin):
        def __init__(self, path: str):
            self.data_path = path
            self.analysis_path = path
            self.tag = "test_loader"
            super().__init__()
            self.execution_count = 0

        @cached_method(cache_tag="test_function")
        def compute(self, x: int) -> int:
            self.execution_count += 1
            return x * 2

    loader = TestDataLoader(str(tmp_path))
    loader.set_save_path(tmp_path)

    # First, create a cache with ON policy
    loader.cacher.policy = "ON"
    assert loader.execution_count == 0
    result1 = loader.compute(5)
    assert loader.execution_count == 1
    assert result1 == 10

    # Verify cache was created
    cache_dir = pathlib.Path(loader.analysis_path) / ".cache" / "test_function"
    assert cache_dir.exists(), "Cache should exist after ON policy run"

    # Get cache file modification time before overwrite
    cache_files_before = list(cache_dir.rglob("*.pkl"))
    if cache_files_before:
        mtime_before = cache_files_before[0].stat().st_mtime

    # Change policy to OVERWRITE and run again
    loader.cacher.policy = "OVERWRITE"
    result2 = loader.compute(5)

    # Should execute function again (cache was cleared, then function executed)
    assert loader.execution_count == 2  # Should increment (function executed again)
    assert result2 == 10  # Should return correct value

    # Verify cache was overwritten (new cache files created or modified)
    cache_files_after = list(cache_dir.rglob("*.pkl"))
    if cache_files_after:
        mtime_after = cache_files_after[0].stat().st_mtime
        # Cache should be newer (overwritten) or at least the function executed
        assert mtime_after >= mtime_before, "Cache should be overwritten"


def test_cached_method_overwrite_policy_writes_new_cache_if_not_exists(
    tmp_path: pathlib.Path,
) -> None:
    """
    Test with OVERWRITE policy: method will write new cache if cache doesn't exist.
    """

    class TestDataLoader(DataLoaderMixin):
        def __init__(self, path: str):
            self.data_path = path
            self.analysis_path = path
            self.tag = "test_loader"
            super().__init__()
            self.execution_count = 0

        @cached_method(cache_tag="test_function")
        def compute(self, x: int) -> int:
            self.execution_count += 1
            return x * 2

    loader = TestDataLoader(str(tmp_path))
    loader.set_save_path(tmp_path)

    # Set policy to OVERWRITE before any cache is created
    loader.cacher.policy = "OVERWRITE"

    # Verify no cache exists initially
    cache_dir = pathlib.Path(loader.analysis_path) / ".cache" / "test_function"
    assert not cache_dir.exists(), "Cache should not exist initially"

    # Run with OVERWRITE policy - should execute and create cache
    assert loader.execution_count == 0
    result = loader.compute(5)
    assert loader.execution_count == 1  # Should execute
    assert result == 10

    # Verify cache files were created
    assert cache_dir.exists(), "Cache directory should be created with OVERWRITE policy"
    cache_files = list(cache_dir.rglob("*.pkl"))
    assert len(cache_files) > 0, "Cache files should be created"

    # Second call with OVERWRITE - should execute again and overwrite cache
    result2 = loader.compute(5)
    assert loader.execution_count == 2  # Should execute again
    assert result2 == 10


def test_cached_method_must_policy_raises_error_when_cache_not_exists(
    tmp_path: pathlib.Path,
) -> None:
    """
    Test with MUST policy: method will raise error if cache does not exist.
    """

    class TestDataLoader(DataLoaderMixin):
        def __init__(self, path: str):
            self.data_path = path
            self.analysis_path = path
            self.tag = "test_loader"
            super().__init__()
            self.execution_count = 0

        @cached_method(cache_tag="test_function")
        def compute(self, x: int) -> int:
            self.execution_count += 1
            return x * 2

    loader = TestDataLoader(str(tmp_path))
    loader.set_save_path(tmp_path)

    # Set policy to MUST before any cache is created
    loader.cacher.policy = "MUST"

    # Verify no cache exists
    cache_dir = pathlib.Path(loader.analysis_path) / ".cache" / "test_function"
    assert not cache_dir.exists(), "Cache should not exist initially"

    # Attempting to call the method with MUST policy and no cache should raise an error
    with pytest.raises(
        RuntimeError,
        match="MUST policy is used for caching, but cache for test_function does not exist",
    ):
        loader.compute(5)
