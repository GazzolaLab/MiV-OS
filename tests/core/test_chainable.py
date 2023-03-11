import pytest
from mock_chain import MockChain

from miv.core.operator.chainable import BaseChainingMixin


def test_chaining_topological_sort():
    a = MockChain(1)
    b = MockChain(2)
    c = MockChain(3)
    d = MockChain(4)
    e = MockChain(5)

    a >> b >> c >> d >> e
    assert a.topological_sort() == [a]
    assert e.topological_sort() == [a, b, c, d, e]

    a.clear_connections()
    assert a.topological_sort() == [a]
    assert b.topological_sort() == [b]
    assert e.topological_sort() == [b, c, d, e]

    a.clear_connections()
    b.clear_connections()
    c.clear_connections()
    d.clear_connections()
    e.clear_connections()

    assert a.topological_sort() == [a]
    assert b.topological_sort() == [b]

    a >> e
    b >> c >> d >> e
    assert a.topological_sort() == [a]
    assert b.topological_sort() == [b]
    assert c.topological_sort() == [b, c]
    assert d.topological_sort() == [b, c, d]
    assert e.topological_sort() == [a, b, c, d, e]


def test_topological_sort_simple_topology():
    a = MockChain(1)
    b = MockChain(2)
    c = MockChain(3)
    d = MockChain(4)
    e = MockChain(5)
    f = MockChain(6)

    # Check connectivity
    a >> b >> c >> d >> e
    f >> c
    assert a.topological_sort() == [a]
    assert e.topological_sort() == [a, b, f, c, d, e]

    # Check cache node
    a.clear_connections()
    c.cacher.value = True
    assert e.topological_sort() == [c, d, e]


def test_topological_sort_loops():
    a = MockChain(1)
    b = MockChain(2)
    c = MockChain(3)
    d = MockChain(4)
    e = MockChain(5)
    f = MockChain(6)

    # Self: a->a
    a >> a
    with pytest.raises(RuntimeError) as e_info:
        a.topological_sort()
        assert "loop" in str(e_info.value)
    a.clear_connections()

    # Circular: a->b->a
    a >> b
    b >> a
    with pytest.raises(RuntimeError) as e_info:
        a.topological_sort()
        assert "loop" in str(e_info.value)
    b.clear_connections()

    # should pass
    a >> c
    a >> b
    b >> c
    assert b.topological_sort() == [a, b]
    assert c.topological_sort() == [a, b, c]

    # should pass
    c >> d >> e
    c >> f >> e
    assert e.topological_sort() == [a, b, c, d, f, e]

    # Circular
    d >> c
    with pytest.raises(RuntimeError) as e_info:
        c.topological_sort()
        assert "loop" in str(e_info.value)


def test_chain_debugging_tools():
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
    a.summarize()
