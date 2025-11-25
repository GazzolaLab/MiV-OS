import os
import pathlib

import numpy as np
import pytest
from joblib import Memory

from miv.core.source.node_mixin import DataLoaderMixin
from miv.core.source.wrapper import cached_method


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
