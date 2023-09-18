import pytest

from miv.core.operator.wrapper import cache_call, cache_functional
from miv.core.operator_generator.wrapper import cache_generator_call


class GeneratorPlotMixin:
    # TODO: This is a temporary solution to avoid issue. calling generator_plot in wrapper should be reconsidered
    def generator_plot(self, *args, **kwargs):
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

    def __init__(self):
        self.cacher = self.MockCacher()
        self.skip_plot = True


class MockObjectWithCache(GeneratorPlotMixin):
    class MockCacher:
        def __init__(self):
            self.cache_dir = ""
            self.cache_tag = "mock_cache"
            self.config_filename = "mock_config"
            self.cache = {}
            self.flag = False

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

    def __init__(self):
        self.cacher = self.MockCacher()
        self.skip_plot = True


@pytest.fixture
def mock_object_without_cache():
    return MockObjectWithoutCache()


@pytest.fixture
def mock_object_with_cache():
    return MockObjectWithCache()


def test_wrap_generator_no_cache(mock_object_without_cache):
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

    a = FooClass()
    assert a(1, 2) == 3
    assert tuple(a.other(bar(), bar())) == (2, 4, 6)


def test_wrap_generator_cache(mock_object_with_cache):
    @cache_generator_call
    def foo(self, x, y):
        return x + y

    def bar():
        yield 1
        yield 2
        yield 3

    assert tuple(foo(mock_object_with_cache, bar(), bar())) == (2, 4, 6)
    for v in mock_object_with_cache.cacher.load_cached():
        assert v == 0  # mock cache only saves zero. (above)

    class FooClass(MockObjectWithCache):
        def __init__(self):
            super().__init__()
            self.called = False

        @cache_generator_call
        def __call__(self, x, y):
            return x + y

        @cache_functional("test1")
        def other(self, x, y):
            if self.called:
                return (
                    -100
                )  # This should not be returned, since cached value is 0 (above)
            self.called = True
            return x + y

    # Test cache_generator_call
    a = FooClass()
    assert tuple(a(bar(), bar())) == (2, 4, 6)
    for v in a.cacher.load_cached():
        assert v == 0  # mock cache only saves zero. (above)

    # Test cache_functional
    a = FooClass()
    assert a.other(1, 5) == 6
    assert a.other(1, 5) != -100
    assert a.other(1, 5) == 0
