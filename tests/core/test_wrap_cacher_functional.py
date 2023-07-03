import os

import numpy as np
import pytest

from miv.core.operator import DataLoaderMixin
from miv.core.operator.cachable import FunctionalCacher
from miv.core.wrapper import wrap_cacher


class Runner:
    def __init__(self, path):
        self.cacher = FunctionalCacher(self)
        self.cacher.cache_dir = path
        self.tag = "test"

        self.run_check_flag = False

    @wrap_cacher(cache_tag="function_1")
    def func1(self, a, b):
        self.run_check_flag = True
        return a + b

    @wrap_cacher(cache_tag="function_2")
    def func2(self, a, b):
        self.run_check_flag = True
        return a - b

    def reset_flag(self):
        self.run_check_flag = False


class MockDataLoader(DataLoaderMixin):
    def __init__(self, path):
        super().__init__()

        self.cacher.cache_dir = path

        self.run_check_flag = False

    @wrap_cacher(cache_tag="function_1")
    def func1(self, a, b):
        self.run_check_flag = True
        return a + b

    @wrap_cacher(cache_tag="function_2")
    def func2(self, a, b):
        self.run_check_flag = True
        return a - b

    def reset_flag(self):
        self.run_check_flag = False


@pytest.mark.parametrize("cls", [Runner, MockDataLoader])
def test_two_function_caching(cls, tmp_path):
    runner = Runner(tmp_path)

    # Case
    a = 1
    b = 2

    assert not runner.run_check_flag
    ans1 = runner.func1(a, b)
    assert runner.run_check_flag
    runner.reset_flag()
    ans2 = runner.func2(a, b)
    assert runner.run_check_flag

    assert ans1 == a + b
    assert ans2 == a - b
    assert ans1 != ans2, "If this fail, change the test cases"

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
    assert not runner.run_check_flag

    assert ans1 == cached_ans1
    assert ans2 == cached_ans2

    print(os.listdir(tmp_path))
    assert os.path.exists(tmp_path / "cache_function_1_0000.pkl")
    assert os.path.exists(tmp_path / "cache_function_2_0000.pkl")
    assert os.path.exists(tmp_path / "config_function_2.json")
    assert os.path.exists(tmp_path / "config_function_1.json")
