import pytest

from tests.core.mock_chain import MockChain, MockChainWithCache, MockChainWithoutCacher
from miv.core.utils.graph_sorting import topological_sort


def test_chaining_topological_sort() -> None:
    a = MockChain(1)
    b = MockChain(2)
    c = MockChain(3)
    d = MockChain(4)
    e = MockChain(5)

    a >> b >> c >> d >> e
    assert topological_sort(a) == [a]
    assert topological_sort(e) == [a, b, c, d, e]

    a.clear_connections()
    assert topological_sort(a) == [a]
    assert topological_sort(b) == [b]
    assert topological_sort(e) == [b, c, d, e]

    a.clear_connections()
    b.clear_connections()
    c.clear_connections()
    d.clear_connections()
    e.clear_connections()

    assert topological_sort(a) == [a]
    assert topological_sort(b) == [b]

    a >> e
    b >> c >> d >> e
    assert topological_sort(a) == [a]
    assert topological_sort(b) == [b]
    assert topological_sort(c) == [b, c]
    assert topological_sort(d) == [b, c, d]
    assert topological_sort(e) == [a, b, c, d, e]


def test_topological_sort_simple_topology() -> None:
    a = MockChain(1)
    b = MockChain(2)
    c = MockChain(3)
    d = MockChain(4)
    e = MockChain(5)
    f = MockChain(6)

    # Check connectivity
    a >> b >> c >> d >> e
    f >> c
    assert topological_sort(a) == [a]
    assert topological_sort(e) == [a, b, f, c, d, e]

    # Check cache node
    a.clear_connections()
    c.cacher.value = True
    assert topological_sort(e) == [c, d, e]


def test_topological_sort_without_cacher() -> None:
    a = MockChain(1)
    b = MockChainWithoutCacher(2)
    c = MockChainWithoutCacher(3)

    # Check connectivity
    a >> b >> c
    assert topological_sort(c) == [a, b, c]


def test_topological_sort_broken_cacher() -> None:
    a = MockChain(1)
    b = MockChain(2)
    c = MockChain(3)

    # topological sort should not be effective if cacher is broken
    a.cacher.check_cached = None
    del a.cacher.check_cached
    del b.cacher

    # Check connectivity
    a >> b >> c
    assert topological_sort(c) == [a, b, c]


def test_topological_sort_cache_skip() -> None:
    a = MockChain(1)
    b = MockChainWithCache(2)
    c = MockChain(3)

    # Check connectivity
    a >> b >> c
    assert topological_sort(a) == [a]
    assert topological_sort(c) == [b, c]


def test_topological_sort_loops() -> None:
    a = MockChain(1)
    b = MockChain(2)
    c = MockChain(3)
    d = MockChain(4)
    e = MockChain(5)
    f = MockChain(6)

    # Self: a->a
    a >> a
    with pytest.raises(RuntimeError) as e_info:
        topological_sort(a)
        assert "loop" in str(e_info.value)
    a.clear_connections()

    # Circular: a->b->a
    a >> b
    b >> a
    with pytest.raises(RuntimeError) as e_info:
        topological_sort(a)
        assert "loop" in str(e_info.value)
    b.clear_connections()

    # should pass
    a >> c
    a >> b
    b >> c
    assert topological_sort(b) == [a, b]
    assert topological_sort(c) == [a, b, c]

    # should pass
    c >> d >> e
    c >> f >> e
    assert topological_sort(e) == [a, b, c, d, f, e]

    # Circular
    d >> c
    with pytest.raises(RuntimeError) as e_info:
        topological_sort(c)
        assert "loop" in str(e_info.value)


def test_chain_debugging_tools() -> None:
    a = MockChain(1)
    b = MockChain(2)
    c = MockChain(3)
    d = MockChain(4)
    e = MockChain(5)
    f = MockChain(6)

    a >> b >> c >> d >> e >> f
    b >> d >> f
    c >> f

    a.visualize()
