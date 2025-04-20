import pytest

from miv.core.operator.wrapper import cache_call, cached_method
from miv.core.operator_generator.wrapper import cache_generator_call


class GeneratorPlotMixin:
    # TODO: This is a temporary solution to avoid issue. calling generator_plot in wrapper should be reconsidered
    def _callback_generator_plot(self, *args, **kwargs):
        pass

    def _callback_firstiter_plot(self, *args, **kwargs):
        pass


class MockObjectWithoutCache(GeneratorPlotMixin):
    class MockCacher:
        def __init__(self):
            self.cache_dir = ""
            self.cache_tag = "mock_cache"
            self.config_filename = "mock_config"

        def cache_filename(self):
            return "mock_cache_file"

        def load_cached(self, tag=None):
            yield 0

        def save_cache(self, data, idx=0, tag=None):
            return True

        def save_config(self, tag=None, *args, **kwargs):
            return None

        def check_cached(self, tag=None, *args, **kwargs):
            return False

    def __init__(self, tmp_path):
        self.cacher = self.MockCacher()
        self.skip_plot = True
        self.analysis_path = tmp_path


class MockObjectWithCache(GeneratorPlotMixin):
    class MockCacher:
        def __init__(self):
            self.cache_dir = ""
            self.cache_tag = "mock_cache"
            self.config_filename = "mock_config"
            self.cache = {}
            self.flag = False
            self.policy = "ON"

        def cache_filename(self):
            return "mock_cache_file"

        def load_cached(self, tag=None):
            for k, v in self.cache.items():
                yield v

        def save_cache(self, data, idx=0, tag=None):
            self.cache[idx] = 0
            self.flag = True
            return True

        def save_config(self, tag=None, *args, **kwargs):
            return None

        def check_cached(self, tag=None, *args, **kwargs):
            return self.flag

    def __init__(self, tmp_path):
        self.cacher = self.MockCacher()
        self.analysis_path = tmp_path
        self.cacher.cache_dir = tmp_path


@pytest.fixture
def mock_object_without_cache(tmp_path):
    return MockObjectWithoutCache(tmp_path)


@pytest.fixture
def mock_object_with_cache(tmp_path):
    return MockObjectWithCache(tmp_path)


def test_wrap_generator_no_cache(mock_object_without_cache, tmp_path):
    @cache_call
    def foo(self, x, y):
        return x + y

    @cache_generator_call
    def foo2(self, x, y):
        return x + y

    def bar():
        yield 1
        yield 2
        yield 3

    assert foo(mock_object_without_cache, 1, 2) == 3
    assert tuple(foo2(mock_object_without_cache, bar(), bar())) == (2, 4, 6)

    class FooClass(MockObjectWithoutCache):
        @cache_call
        def __call__(self, x, y):
            return x + y

        @cache_generator_call
        def other(self, x, y):
            return x + y

    a = FooClass(tmp_path)
    assert a(1, 2) == 3
    assert tuple(a.other(bar(), bar())) == (2, 4, 6)
