import pytest

from miv.core.wrapper import wrap_generator_to_generator


class MockObjectWithoutCache:
    class MockCacher:
        def __init__(self):
            self.cache_dir = ""
            self.cache_tag = "mock_cache"
            self.config_filename = "mock_config"

        def cache_filename(self):
            return "mock_cache_file"

        def load_cached(self):
            yield 0

        def save_cache(self, data, idx=0):
            return True

        def save_config(self):
            return None

        def check_cached(self):
            return False

    def __init__(self):
        self.cacher = self.MockCacher()


class MockObjectWithCache:
    class MockCacher:
        def __init__(self):
            self.cache_dir = ""
            self.cache_tag = "mock_cache"
            self.config_filename = "mock_config"
            self.cache = {}
            self.flag = False

        def cache_filename(self):
            return "mock_cache_file"

        def load_cached(self):
            for k, v in self.cache.items():
                yield v

        def save_cache(self, data, idx=0):
            self.cache[idx] = 0
            self.flag = True
            return True

        def save_config(self):
            return None

        def check_cached(self):
            return self.flag

    def __init__(self):
        self.cacher = self.MockCacher()


@pytest.fixture
def mock_object_without_cache():
    return MockObjectWithoutCache()


@pytest.fixture
def mock_object_with_cache():
    return MockObjectWithCache()


def test_wrap_generator_no_cache(mock_object_without_cache):
    @wrap_generator_to_generator
    def foo(self, x, y):
        return x + y

    def bar():
        yield 1
        yield 2
        yield 3

    assert foo(mock_object_without_cache, 1, 2) == 3
    assert tuple(foo(mock_object_without_cache, bar(), bar())) == (2, 4, 6)

    class FooClass(MockObjectWithoutCache):
        @wrap_generator_to_generator
        def __call__(self, x, y):
            return x + y

        @wrap_generator_to_generator
        def other(self, x, y):
            return x + y

    a = FooClass()
    assert a.other(1, 2) == 3
    assert a(1, 2) == 3
    assert tuple(a.other(bar(), bar())) == (2, 4, 6)
    assert tuple(a(bar(), bar())) == (2, 4, 6)


def test_wrap_generator_cache(mock_object_with_cache):
    @wrap_generator_to_generator
    def foo(self, x, y):
        return x + y

    def bar():
        yield 1
        yield 2
        yield 3

    assert tuple(foo(mock_object_with_cache, bar(), bar())) == (2, 4, 6)
    assert tuple(foo(mock_object_with_cache, bar(), bar())) == (
        0,
        0,
        0,
    )  # mock cache only saves zero. (above)

    class FooClass(MockObjectWithCache):
        @wrap_generator_to_generator
        def __call__(self, x, y):
            return x + y

        @wrap_generator_to_generator
        def other(self, x, y):
            return x + y

    a = FooClass()
    assert tuple(a.other(bar(), bar())) == (2, 4, 6)
    assert tuple(a.other(bar(), bar())) == (0, 0, 0)
    assert tuple(a(bar(), bar())) == (0, 0, 0)  # Sample class share cache
    assert tuple(a(bar(), bar())) == (0, 0, 0)
