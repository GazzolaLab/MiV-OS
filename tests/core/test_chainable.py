import pytest

from tests.core.mock_chain import MockChain, MockChainWithCache, MockChainWithoutCacher
from miv.core.utils.graph_sorting import topological_sort
from miv.core.chainable import node_graph_visualize
import matplotlib.pyplot as plt
import networkx as nx


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
    assert hasattr(iterator, "__iter__")
    assert hasattr(iterator, "__next__")

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
    assert hasattr(iterator, "__iter__")
    assert hasattr(iterator, "__next__")

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
    assert hasattr(iterator, "__iter__")
    assert hasattr(iterator, "__next__")

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
    assert hasattr(iterator, "__iter__")
    assert hasattr(iterator, "__next__")

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


def test_disconnect_downstream_removes_node_from_downstream_list() -> None:
    """Test that disconnect_downstream() removes a node from the downstream list."""
    node = MockChain(1)
    downstream_node1 = MockChain(2)
    downstream_node2 = MockChain(3)
    downstream_node3 = MockChain(4)

    # Add multiple downstream nodes
    node.append_downstream(downstream_node1)
    node.append_downstream(downstream_node2)
    node.append_downstream(downstream_node3)

    # Verify all nodes are in downstream list
    assert len(node._downstream_list) == 3
    assert downstream_node1 in node._downstream_list
    assert downstream_node2 in node._downstream_list
    assert downstream_node3 in node._downstream_list

    # Disconnect middle node
    node.disconnect_downstream(downstream_node2)

    # Verify downstream_node2 is removed
    assert len(node._downstream_list) == 2
    assert downstream_node1 in node._downstream_list
    assert downstream_node2 not in node._downstream_list
    assert downstream_node3 in node._downstream_list

    # Verify order is preserved for remaining nodes
    assert node._downstream_list[0] is downstream_node1
    assert node._downstream_list[1] is downstream_node3

    # Disconnect first node
    node.disconnect_downstream(downstream_node1)

    # Verify downstream_node1 is removed
    assert len(node._downstream_list) == 1
    assert downstream_node1 not in node._downstream_list
    assert downstream_node3 in node._downstream_list

    # Disconnect last node
    node.disconnect_downstream(downstream_node3)

    # Verify downstream list is now empty
    assert len(node._downstream_list) == 0
    assert downstream_node3 not in node._downstream_list

    # Verify upstream list is not affected
    assert node._upstream_list == []


def test_disconnect_upstream_raises_error_if_node_not_in_upstream_list() -> None:
    """Test that disconnect_upstream() raises ValueError if node not in upstream list."""
    node = MockChain(1)
    upstream_node1 = MockChain(2)
    upstream_node2 = MockChain(3)
    unrelated_node = MockChain(4)

    # Add some upstream nodes
    node.append_upstream(upstream_node1)
    node.append_upstream(upstream_node2)

    # Verify unrelated_node is not in upstream list
    assert unrelated_node not in node._upstream_list

    # Attempting to disconnect a node not in upstream list should raise ValueError
    with pytest.raises(ValueError):
        node.disconnect_upstream(unrelated_node)

    # Verify upstream list is unchanged
    assert len(node._upstream_list) == 2
    assert upstream_node1 in node._upstream_list
    assert upstream_node2 in node._upstream_list

    # Test with empty upstream list
    node2 = MockChain(5)
    assert len(node2._upstream_list) == 0

    # Attempting to disconnect from empty list should raise ValueError
    with pytest.raises(ValueError):
        node2.disconnect_upstream(upstream_node1)


def test_disconnect_downstream_raises_error_if_node_not_in_downstream_list() -> None:
    """Test that disconnect_downstream() raises ValueError if node not in downstream list."""
    node = MockChain(1)
    downstream_node1 = MockChain(2)
    downstream_node2 = MockChain(3)
    unrelated_node = MockChain(4)

    # Add some downstream nodes
    node.append_downstream(downstream_node1)
    node.append_downstream(downstream_node2)

    # Verify unrelated_node is not in downstream list
    assert unrelated_node not in node._downstream_list

    # Attempting to disconnect a node not in downstream list should raise ValueError
    with pytest.raises(ValueError):
        node.disconnect_downstream(unrelated_node)

    # Verify downstream list is unchanged
    assert len(node._downstream_list) == 2
    assert downstream_node1 in node._downstream_list
    assert downstream_node2 in node._downstream_list

    # Test with empty downstream list
    node2 = MockChain(5)
    assert len(node2._downstream_list) == 0

    # Attempting to disconnect from empty list should raise ValueError
    with pytest.raises(ValueError):
        node2.disconnect_downstream(downstream_node1)


def test_clear_connections_removes_all_upstream_and_downstream_connections() -> None:
    """Test that clear_connections() removes all upstream and downstream connections."""
    node = MockChain(1)
    upstream_node1 = MockChain(2)
    upstream_node2 = MockChain(3)
    downstream_node1 = MockChain(4)
    downstream_node2 = MockChain(5)

    # Set up bidirectional connections using >> operator
    upstream_node1 >> node
    upstream_node2 >> node
    node >> downstream_node1
    node >> downstream_node2

    # Verify connections exist
    assert len(node._upstream_list) == 2
    assert len(node._downstream_list) == 2
    assert upstream_node1 in node._upstream_list
    assert upstream_node2 in node._upstream_list
    assert downstream_node1 in node._downstream_list
    assert downstream_node2 in node._downstream_list

    # Verify bidirectional relationships exist before clearing
    assert node in upstream_node1._downstream_list
    assert node in upstream_node2._downstream_list
    assert node in downstream_node1._upstream_list
    assert node in downstream_node2._upstream_list

    # Clear connections
    node.clear_connections()

    # Verify all connections are removed from node
    assert len(node._upstream_list) == 0
    assert len(node._downstream_list) == 0
    assert upstream_node1 not in node._upstream_list
    assert upstream_node2 not in node._upstream_list
    assert downstream_node1 not in node._downstream_list
    assert downstream_node2 not in node._downstream_list

    # Verify bidirectional relationships are also removed (node removed from other nodes' lists)
    assert node not in upstream_node1._downstream_list
    assert node not in upstream_node2._downstream_list
    assert node not in downstream_node1._upstream_list
    assert node not in downstream_node2._upstream_list


def test_clear_connections_works_on_node_with_no_connections() -> None:
    """Test that clear_connections() works on node with no connections."""
    node = MockChain(1)

    # Verify node has no connections
    assert len(node._upstream_list) == 0
    assert len(node._downstream_list) == 0

    # Clear connections should not raise an error
    node.clear_connections()

    # Verify node still has no connections
    assert len(node._upstream_list) == 0
    assert len(node._downstream_list) == 0


def test_visualize_creates_networkx_digraph_and_returns_it() -> None:
    """Test that visualize() creates NetworkX DiGraph from node structure and returns it."""
    node = MockChain(1)
    downstream_node1 = MockChain(2)
    downstream_node2 = MockChain(3)

    # Set up connections
    node >> downstream_node1
    node >> downstream_node2

    # Create matplotlib axes
    fig, ax = plt.subplots()

    # Call visualize
    G = node.visualize(ax)

    # Verify it returns a NetworkX DiGraph
    assert isinstance(G, nx.DiGraph)
    assert G.is_directed()

    # Verify the graph contains nodes
    assert len(G.nodes()) > 0

    # Verify the graph structure matches the connections
    node_repr = repr(node)
    downstream1_repr = repr(downstream_node1)
    downstream2_repr = repr(downstream_node2)

    assert node_repr in G.nodes()
    assert downstream1_repr in G.nodes()
    assert downstream2_repr in G.nodes()

    # Verify edges exist
    assert G.has_edge(node_repr, downstream1_repr)
    assert G.has_edge(node_repr, downstream2_repr)


def test_visualize_traverses_all_downstream_nodes_using_bfs() -> None:
    """Test that visualize() traverses all downstream nodes using BFS (breadth-first search)."""
    # Create a graph structure where BFS and DFS would visit nodes in different orders
    # Structure:
    #   a -> b -> d
    #   a -> c -> e
    # BFS order: a, b, c, d, e (level by level)
    # DFS order: a, b, d, c, e (or a, c, e, b, d)
    a = MockChain(1)
    b = MockChain(2)
    c = MockChain(3)
    d = MockChain(4)
    e = MockChain(5)

    a >> b >> d
    a >> c >> e

    # Create matplotlib axes
    fig, ax = plt.subplots()

    # Call visualize
    G = a.visualize(ax)
    plt.close(fig)
    # Verify all nodes are in the graph (BFS should visit all reachable nodes)
    assert repr(a) in G.nodes()
    assert repr(b) in G.nodes()
    assert repr(c) in G.nodes()
    assert repr(d) in G.nodes()
    assert repr(e) in G.nodes()

    # Verify all edges are present
    assert G.has_edge(repr(a), repr(b))
    assert G.has_edge(repr(a), repr(c))
    assert G.has_edge(repr(b), repr(d))
    assert G.has_edge(repr(c), repr(e))

    assert len(G.nodes()) == 5  # All 5 nodes should be in the graph
    assert len(G.edges()) == 4  # All 4 edges should be in the graph
    plt.close(fig)


def test_visualize_uses_provided_matplotlib_axes_for_drawing() -> None:
    """Test that visualize() uses the provided matplotlib Axes for drawing."""
    node = MockChain(1)
    downstream_node1 = MockChain(2)
    downstream_node2 = MockChain(3)

    # Set up connections
    node >> downstream_node1
    node >> downstream_node2

    # Create matplotlib axes
    fig, ax = plt.subplots()

    # Verify axes is initially empty (no children)
    initial_children_count = len(ax.get_children())

    # Call visualize
    G = node.visualize(ax)

    # Verify axes has been used for drawing
    # After visualization, axes should have children (artists drawn on it)
    final_children_count = len(ax.get_children())
    assert final_children_count > initial_children_count

    # Verify axes properties are set by visualize
    # The visualize function sets ax.margins(x=0.4) and ax.axis("off")
    # We can check that the axis is turned off
    assert not ax.axison  # ax.axis("off") sets axison to False

    # Verify the axes is the same one we passed in
    assert ax is not None
    assert isinstance(ax, plt.Axes)

    plt.close(fig)


def test_node_graph_visualize_standalone_function_works_independently_of_mixin() -> (
    None
):
    """Test that node_graph_visualize() standalone function works independently of mixin."""
    # Create nodes without using the mixin's visualize method
    node = MockChain(1)
    downstream_node1 = MockChain(2)
    downstream_node2 = MockChain(3)

    # Set up connections
    node >> downstream_node1
    node >> downstream_node2

    # Create matplotlib axes
    fig, ax = plt.subplots()

    # Call node_graph_visualize directly (not through mixin's visualize method)
    G = node_graph_visualize(ax, node)

    # Verify it returns a NetworkX DiGraph
    assert isinstance(G, nx.DiGraph)
    assert G.is_directed()

    # Verify the graph contains nodes
    assert len(G.nodes()) > 0

    # Verify the graph structure matches the connections
    node_repr = repr(node)
    downstream1_repr = repr(downstream_node1)
    downstream2_repr = repr(downstream_node2)

    assert node_repr in G.nodes()
    assert downstream1_repr in G.nodes()
    assert downstream2_repr in G.nodes()

    # Verify edges exist
    assert G.has_edge(node_repr, downstream1_repr)
    assert G.has_edge(node_repr, downstream2_repr)

    # Verify axes has been used for drawing
    assert not ax.axison  # ax.axis("off") sets axison to False

    plt.close(fig)


def test_node_graph_visualize_accepts_start_node_parameter() -> None:
    """Test that node_graph_visualize() accepts start_node parameter."""
    # Create a more complex graph
    a = MockChain(1)
    b = MockChain(2)
    c = MockChain(3)
    d = MockChain(4)

    a >> b >> d
    a >> c >> d

    # Create matplotlib axes
    fig, ax = plt.subplots()

    # Call node_graph_visualize with different start nodes
    # Starting from 'a' should include all nodes
    G_from_a = node_graph_visualize(ax, a)
    assert repr(a) in G_from_a.nodes()
    assert repr(b) in G_from_a.nodes()
    assert repr(c) in G_from_a.nodes()
    assert repr(d) in G_from_a.nodes()

    # Starting from 'b' should only include b and d
    fig2, ax2 = plt.subplots()
    G_from_b = node_graph_visualize(ax2, b)
    assert repr(b) in G_from_b.nodes()
    assert repr(d) in G_from_b.nodes()
    assert repr(a) not in G_from_b.nodes()
    assert repr(c) not in G_from_b.nodes()

    plt.close(fig)
    plt.close(fig2)


def test_text_visualize_hierarchy_formats_hierarchy_with_proper_indentation() -> None:
    """Test that text_visualize_hierarchy() formats hierarchy with proper indentation."""
    node = MockChain(1)
    child1 = MockChain(2)
    child2 = MockChain(3)
    grandchild = MockChain(4)

    # Create a hierarchy: node -> child1 -> grandchild, node -> child2
    string_list = [
        (0, node),  # Root node
        (1, child1),  # First level child
        (2, grandchild),  # Second level child
        (1, child2),  # Another first level child
    ]

    result = node.text_visualize_hierarchy(string_list)

    # Verify it returns a string
    assert isinstance(result, str)

    # Verify it contains all nodes
    assert str(node) in result
    assert str(child1) in result
    assert str(child2) in result
    assert str(grandchild) in result

    # Verify proper indentation (lines should be separated)
    lines = result.split("\n")
    assert len(lines) == 4  # Should have 4 lines for 4 nodes


def test_text_visualize_hierarchy_uses_prefix_for_nodes_with_depth_greater_than_zero() -> (
    None
):
    """Test that text_visualize_hierarchy() uses prefix for nodes with depth > 0."""
    node = MockChain(1)
    child = MockChain(2)

    string_list = [
        (0, node),
        (1, child),
    ]

    result = node.text_visualize_hierarchy(string_list, prefix="|__ ")

    lines = result.split("\n")
    # Root node (depth 0) should not have prefix
    assert lines[0] == str(node)
    # Child node (depth > 0) should have prefix
    assert lines[1].startswith("|__ ")
    assert str(child) in lines[1]


def test_text_visualize_hierarchy_does_not_add_prefix_for_root_node_depth_zero() -> (
    None
):
    """Test that text_visualize_hierarchy() does not add prefix for root node (depth 0)."""
    node = MockChain(1)
    child = MockChain(2)

    string_list = [
        (0, node),
        (1, child),
    ]

    result = node.text_visualize_hierarchy(string_list, prefix="|__ ")

    lines = result.split("\n")
    # Root node should be exactly the string representation, no prefix
    assert lines[0] == str(node)
    assert not lines[0].startswith("|__ ")


def test_text_visualize_hierarchy_handles_empty_string_list() -> None:
    """Test that text_visualize_hierarchy() handles empty string_list."""
    node = MockChain(1)

    string_list: list[tuple[int, MockChain]] = []

    result = node.text_visualize_hierarchy(string_list)

    # Should return empty string or newline-only string
    assert isinstance(result, str)
    # Empty list should result in empty string (or just newlines)
    lines = result.split("\n")
    # If empty, should have at most one empty line
    assert len([line for line in lines if line.strip()]) == 0


def test_flow_blocked_should_be_abstract_method() -> None:
    """Test that flow_blocked() should be abstract method."""
    from miv.core.chainable import ChainingMixin

    # Create a class that inherits from ChainingMixin but doesn't implement flow_blocked
    class IncompleteChain(ChainingMixin):
        def __init__(self):
            super().__init__()

    # Attempting to instantiate should raise TypeError because abstract method is not implemented
    with pytest.raises(
        TypeError, match="Can't instantiate abstract class IncompleteChain"
    ):
        IncompleteChain()


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

    # Check cache node - when c is flow-blocked, upstream nodes are skipped
    a.clear_connections()
    c._flow_blocked = True
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

    # (MockChain no longer uses cacher, flow_blocked() uses _flow_blocked instead)
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
