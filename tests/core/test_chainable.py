import pytest

from tests.core.mock_chain import MockChain, MockChainWithCache, MockChainWithoutCacher
from miv.core.utils.graph_sorting import topological_sort
import matplotlib.pyplot as plt


def test_chaining_mixin_init_initializes_empty_lists() -> None:
    """Test that ChainingMixin.__init__() initializes empty upstream and downstream lists."""
    node = MockChain(1)
    assert node._upstream_list == []
    assert node._downstream_list == []
    assert len(node._upstream_list) == 0
    assert len(node._downstream_list) == 0


def test_rshift_operator_appends_bidirectional_relationship() -> None:
    """Test that __rshift__() operator (>>) correctly sets up bidirectional relationship."""
    left = MockChain(1)
    right = MockChain(2)

    # Before chaining, both should have empty lists
    assert left._downstream_list == []
    assert left._upstream_list == []
    assert right._downstream_list == []
    assert right._upstream_list == []

    # Chain them using >> operator
    result = left >> right

    # Verify right node is in left's downstream list
    assert right in left._downstream_list
    assert len(left._downstream_list) == 1

    # Verify left node is in right's upstream list
    assert left in right._upstream_list
    assert len(right._upstream_list) == 1

    # Verify left's upstream and right's downstream remain empty
    assert left._upstream_list == []
    assert right._downstream_list == []

    # Verify __rshift__ returns the right node for method chaining
    assert result is right


def test_rshift_operator_returns_right_node_for_method_chaining() -> None:
    """Test that __rshift__() returns the right node to enable method chaining."""
    a = MockChain(1)
    b = MockChain(2)
    c = MockChain(3)

    # Method chaining should work: a >> b >> c
    # This requires that (a >> b) returns b, so we can then do (b >> c)
    result = a >> b >> c

    # Verify the final result is c (the rightmost node)
    assert result is c

    # Verify the chain was created correctly
    assert b in a._downstream_list
    assert c in b._downstream_list
    assert a in b._upstream_list
    assert b in c._upstream_list

    # Verify intermediate chaining also returns correct node
    x = MockChain(4)
    y = MockChain(5)
    intermediate = x >> y
    assert intermediate is y


def test_append_upstream_adds_node_to_upstream_list() -> None:
    """Test that append_upstream() adds a node to the upstream list."""
    node = MockChain(1)
    upstream_node = MockChain(2)

    # Initially, upstream list should be empty
    assert node._upstream_list == []
    assert len(node._upstream_list) == 0

    # Append upstream node
    node.append_upstream(upstream_node)

    # Verify upstream node is in the list
    assert upstream_node in node._upstream_list
    assert len(node._upstream_list) == 1
    assert node._upstream_list[0] is upstream_node

    # Verify downstream list is not affected
    assert node._downstream_list == []


def test_append_downstream_adds_node_to_downstream_list() -> None:
    """Test that append_downstream() adds a node to the downstream list."""
    node = MockChain(1)
    downstream_node = MockChain(2)

    # Initially, downstream list should be empty
    assert node._downstream_list == []
    assert len(node._downstream_list) == 0

    # Append downstream node
    node.append_downstream(downstream_node)

    # Verify downstream node is in the list
    assert downstream_node in node._downstream_list
    assert len(node._downstream_list) == 1
    assert node._downstream_list[0] is downstream_node

    # Verify upstream list is not affected
    assert node._upstream_list == []


def test_append_upstream_allows_multiple_upstream_nodes() -> None:
    """Test that append_upstream() can be called multiple times to add multiple upstream nodes."""
    node = MockChain(1)
    upstream_node1 = MockChain(2)
    upstream_node2 = MockChain(3)
    upstream_node3 = MockChain(4)

    # Initially, upstream list should be empty
    assert node._upstream_list == []
    assert len(node._upstream_list) == 0

    # Append first upstream node
    node.append_upstream(upstream_node1)
    assert len(node._upstream_list) == 1
    assert upstream_node1 in node._upstream_list

    # Append second upstream node
    node.append_upstream(upstream_node2)
    assert len(node._upstream_list) == 2
    assert upstream_node1 in node._upstream_list
    assert upstream_node2 in node._upstream_list

    # Append third upstream node
    node.append_upstream(upstream_node3)
    assert len(node._upstream_list) == 3
    assert upstream_node1 in node._upstream_list
    assert upstream_node2 in node._upstream_list
    assert upstream_node3 in node._upstream_list

    # Verify order is preserved (nodes added in order)
    assert node._upstream_list[0] is upstream_node1
    assert node._upstream_list[1] is upstream_node2
    assert node._upstream_list[2] is upstream_node3

    # Verify downstream list is not affected
    assert node._downstream_list == []


def test_append_downstream_allows_multiple_downstream_nodes() -> None:
    """Test that append_downstream() can be called multiple times to add multiple downstream nodes."""
    node = MockChain(1)
    downstream_node1 = MockChain(2)
    downstream_node2 = MockChain(3)
    downstream_node3 = MockChain(4)

    # Initially, downstream list should be empty
    assert node._downstream_list == []
    assert len(node._downstream_list) == 0

    # Append first downstream node
    node.append_downstream(downstream_node1)
    assert len(node._downstream_list) == 1
    assert downstream_node1 in node._downstream_list

    # Append second downstream node
    node.append_downstream(downstream_node2)
    assert len(node._downstream_list) == 2
    assert downstream_node1 in node._downstream_list
    assert downstream_node2 in node._downstream_list

    # Append third downstream node
    node.append_downstream(downstream_node3)
    assert len(node._downstream_list) == 3
    assert downstream_node1 in node._downstream_list
    assert downstream_node2 in node._downstream_list
    assert downstream_node3 in node._downstream_list

    # Verify order is preserved (nodes added in order)
    assert node._downstream_list[0] is downstream_node1
    assert node._downstream_list[1] is downstream_node2
    assert node._downstream_list[2] is downstream_node3

    # Verify upstream list is not affected
    assert node._upstream_list == []


def test_iterate_upstream_returns_iterator_over_all_upstream_nodes() -> None:
    """Test that iterate_upstream() returns an iterator over all upstream nodes."""
    node = MockChain(1)
    upstream_node1 = MockChain(2)
    upstream_node2 = MockChain(3)

    # Add multiple upstream nodes
    node.append_upstream(upstream_node1)
    node.append_upstream(upstream_node2)

    # Get iterator
    iterator = node.iterate_upstream()

    # Verify it's an iterator
    assert hasattr(iterator, '__iter__')
    assert hasattr(iterator, '__next__')

    # Verify it iterates over all upstream nodes
    nodes_from_iterator = list(iterator)
    assert len(nodes_from_iterator) == 2
    assert upstream_node1 in nodes_from_iterator
    assert upstream_node2 in nodes_from_iterator

    # Verify order is preserved
    assert nodes_from_iterator[0] is upstream_node1
    assert nodes_from_iterator[1] is upstream_node2


def test_iterate_upstream_returns_empty_iterator_when_no_upstream_nodes() -> None:
    """Test that iterate_upstream() returns empty iterator when no upstream nodes exist."""
    node = MockChain(1)

    # Verify node has no upstream nodes initially
    assert len(node._upstream_list) == 0

    # Get iterator
    iterator = node.iterate_upstream()

    # Verify it's an iterator
    assert hasattr(iterator, '__iter__')
    assert hasattr(iterator, '__next__')

    # Verify it returns empty iterator
    nodes_from_iterator = list(iterator)
    assert len(nodes_from_iterator) == 0
    assert nodes_from_iterator == []

    # Verify calling next() raises StopIteration
    empty_iterator = node.iterate_upstream()
    with pytest.raises(StopIteration):
        next(empty_iterator)


def test_iterate_downstream_returns_iterator_over_all_downstream_nodes() -> None:
    """Test that iterate_downstream() returns an iterator over all downstream nodes."""
    node = MockChain(1)
    downstream_node1 = MockChain(2)
    downstream_node2 = MockChain(3)

    # Add multiple downstream nodes
    node.append_downstream(downstream_node1)
    node.append_downstream(downstream_node2)

    # Get iterator
    iterator = node.iterate_downstream()

    # Verify it's an iterator
    assert hasattr(iterator, '__iter__')
    assert hasattr(iterator, '__next__')

    # Verify it iterates over all downstream nodes
    nodes_from_iterator = list(iterator)
    assert len(nodes_from_iterator) == 2
    assert downstream_node1 in nodes_from_iterator
    assert downstream_node2 in nodes_from_iterator

    # Verify order is preserved
    assert nodes_from_iterator[0] is downstream_node1
    assert nodes_from_iterator[1] is downstream_node2


def test_iterate_downstream_returns_empty_iterator_when_no_downstream_nodes() -> None:
    """Test that iterate_downstream() returns empty iterator when no downstream nodes exist."""
    node = MockChain(1)

    # Verify node has no downstream nodes initially
    assert len(node._downstream_list) == 0

    # Get iterator
    iterator = node.iterate_downstream()

    # Verify it's an iterator
    assert hasattr(iterator, '__iter__')
    assert hasattr(iterator, '__next__')

    # Verify it returns empty iterator
    nodes_from_iterator = list(iterator)
    assert len(nodes_from_iterator) == 0
    assert nodes_from_iterator == []

    # Verify calling next() raises StopIteration
    empty_iterator = node.iterate_downstream()
    with pytest.raises(StopIteration):
        next(empty_iterator)


def test_disconnect_upstream_removes_node_from_upstream_list() -> None:
    """Test that disconnect_upstream() removes a node from the upstream list."""
    node = MockChain(1)
    upstream_node1 = MockChain(2)
    upstream_node2 = MockChain(3)
    upstream_node3 = MockChain(4)

    # Add multiple upstream nodes
    node.append_upstream(upstream_node1)
    node.append_upstream(upstream_node2)
    node.append_upstream(upstream_node3)

    # Verify all nodes are in upstream list
    assert len(node._upstream_list) == 3
    assert upstream_node1 in node._upstream_list
    assert upstream_node2 in node._upstream_list
    assert upstream_node3 in node._upstream_list

    # Disconnect middle node
    node.disconnect_upstream(upstream_node2)

    # Verify upstream_node2 is removed
    assert len(node._upstream_list) == 2
    assert upstream_node1 in node._upstream_list
    assert upstream_node2 not in node._upstream_list
    assert upstream_node3 in node._upstream_list

    # Verify order is preserved for remaining nodes
    assert node._upstream_list[0] is upstream_node1
    assert node._upstream_list[1] is upstream_node3

    # Disconnect first node
    node.disconnect_upstream(upstream_node1)

    # Verify upstream_node1 is removed
    assert len(node._upstream_list) == 1
    assert upstream_node1 not in node._upstream_list
    assert upstream_node3 in node._upstream_list

    # Disconnect last node
    node.disconnect_upstream(upstream_node3)

    # Verify upstream list is now empty
    assert len(node._upstream_list) == 0
    assert upstream_node3 not in node._upstream_list

    # Verify downstream list is not affected
    assert node._downstream_list == []


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

    fig, ax = plt.subplots()
    G = a.visualize(ax)
