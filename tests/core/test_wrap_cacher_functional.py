import os
import pathlib

import numpy as np
import pytest

from miv.core.operator.operator import DataLoaderMixin
from miv.core.operator.wrapper import cache_functional


class MockDataLoader(DataLoaderMixin):
    def __init__(self, path):
        self.data_path = path
        super().__init__()
        self.run_check_flag = False
        self.tag = "mock module"

    @cache_functional(cache_tag="function_1")
    def func1(self, a, b):
        self.run_check_flag = True
        return a + b

    @cache_functional(cache_tag="function_2")
    def func2(self, a, b):
        self.run_check_flag = True
        return a - b

    @cache_functional(cache_tag="function_3")
    def func3(self, a, b):
        self.run_check_flag = True
        # This part is to test if the cache is working properly in nested case
        self.func1(a, b)
        return a * b

    def reset_flag(self):
        self.run_check_flag = False


@pytest.mark.mpi_xfail
@pytest.mark.parametrize("cls", [MockDataLoader])
def test_two_function_caching(cls, tmp_path):
    runner = cls(tmp_path)
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

    analysis_path = pathlib.Path(runner.analysis_path)
    print(os.listdir(analysis_path))
    print(os.listdir(analysis_path / ".cache"))
    assert os.path.exists(
        analysis_path / ".cache" / "cache_function_1_rank000_0000.pkl"
    )
    assert os.path.exists(
        analysis_path / ".cache" / "cache_function_2_rank000_0000.pkl"
    )
    assert os.path.exists(analysis_path / ".cache" / "config_function_2.json")
    assert os.path.exists(analysis_path / ".cache" / "config_function_1.json")


@pytest.mark.parametrize("cls", [MockDataLoader])
def test_two_function_caching_cacher_called_flag_test(cls, tmp_path, mocker):
    runner = cls(tmp_path)

    # Case
    a = 1
    b = 2

    spy_load = mocker.spy(runner.cacher, "load_cached")
    spy_save = mocker.spy(runner.cacher, "save_cache")

    assert not runner.run_check_flag
    ans1 = runner.func1(a, b)
    assert runner.run_check_flag
    assert spy_load.call_count == 0
    assert spy_save.call_count == 1
    runner.reset_flag()
    ans2 = runner.func2(a, b)
    assert runner.run_check_flag
    assert spy_load.call_count == 0
    assert spy_save.call_count == 2
    runner.reset_flag()
    ans3 = runner.func3(a, b)
    assert runner.run_check_flag
    assert spy_load.call_count == 1  # This is because func1 is called inside func3
    assert spy_save.call_count == 3

    assert ans1 == a + b
    assert ans2 == a - b
    assert ans3 == a * b
    assert ans1 != ans2, "If this fail, change the test cases"
    assert ans1 != ans3, "If this fail, change the test cases"

    # Check for multiple run
    runner.reset_flag()
    cached_ans1 = runner.func1(a, b)
    assert cached_ans1 == ans1
    assert spy_load.call_count == 2
    assert spy_save.call_count == 3
    cached_ans2 = runner.func2(a, b)
    assert cached_ans2 == ans2
    assert spy_load.call_count == 3
    assert spy_save.call_count == 3
    cached_ans3 = runner.func3(a, b)
    assert cached_ans3 == ans3
    assert spy_load.call_count == 4
    assert spy_save.call_count == 3
    assert not runner.run_check_flag

    assert ans1 == cached_ans1
    assert ans2 == cached_ans2
