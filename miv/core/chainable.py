from __future__ import annotations

__doc__ = """
Mixin to create chaining structure between objects.
"""
__all__ = ["ChainingMixin", "node_graph_visualize"]

from typing import TYPE_CHECKING, Any
from collections.abc import Iterator

import itertools

import matplotlib.pyplot as plt


if TYPE_CHECKING:
    import networkx as nx
    from .protocol import _Chainable


class ChainingMixin:
    """
    Base mixin to create chaining structure between objects.

    Need further implementation of: output, tag
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._downstream_list: list[_Chainable] = []
        self._upstream_list: list[_Chainable] = []

    def __rshift__(self, right: _Chainable) -> _Chainable:
        self.append_downstream(right)
        right.append_upstream(self)
        return right

    def clear_connections(self) -> None:
        """Clear all the connections to other nodes, and remove dependencies."""
        for node in self.iterate_downstream():
            node.disconnect_upstream(self)
        for node in self.iterate_upstream():
            node.disconnect_downstream(self)
        self._downstream_list.clear()
        self._upstream_list.clear()

    def disconnect_upstream(self, node: _Chainable) -> None:
        self._upstream_list.remove(node)

    def disconnect_downstream(self, node: _Chainable) -> None:
        self._downstream_list.remove(node)

    def append_upstream(self, node: _Chainable) -> None:
        self._upstream_list.append(node)

    def append_downstream(self, node: _Chainable) -> None:
        self._downstream_list.append(node)

    def iterate_downstream(self) -> Iterator[_Chainable]:
        return iter(self._downstream_list)

    def iterate_upstream(self) -> Iterator[_Chainable]:
        return iter(self._upstream_list)

    def visualize(self, ax: "plt.Axes", seed: int = 200) -> "nx.DiGraph":
        """
        Visualize the network structure of the "Operator".
        """
        return node_graph_visualize(ax, self, seed)

    def text_visualize_hierarchy(
        self,
        string_list: list[tuple[int, _Chainable]],
        prefix: str = "|__ ",
    ) -> str:
        """
        Generate a text-based visualization of a hierarchical structure.

        Parameters
        ----------
        string_list : list of (int, _Chainable)
            List of (depth, node) tuples representing the hierarchical order and tree depth.
        prefix : str, optional
            Prefix displayed before each node when depth > 0, by default "|__ "

        Returns
        -------
        str
            Multi-line string illustrating the hierarchy with indentation.
        """
        output = []
        for _i, item in enumerate(string_list):
            depth, label = item
            if depth > 0:
                output.append("    " * (depth - 1) + prefix + str(label))
            else:
                output.append(str(label))
        return "\n".join(output)

def node_graph_visualize(ax: "plt.Axes", start_node: _Chainable, seed: int = 200) -> "nx.DiGraph":
    """
    Visualize a node graph starting from a given node.

    Parameters
    ----------
    ax : plt.Axes
        The axes to plot the graph on.
    start_node : _Chainable
        The node to start the graph from.
    seed : int, optional
        The seed for the random number generator.

    Returns
    """

    import networkx as nx
    G = nx.DiGraph()

    # BFS
    visited: list[_Chainable] = []
    next_list: list[_Chainable] = [start_node]
    while next_list:
        v = next_list.pop()
        visited.append(v)
        for node in itertools.chain(v.iterate_downstream()):
            G.add_edge(repr(v), repr(node))
            if node in visited or node in next_list:
                continue
            visited.append(node)
            next_list.append(node)

    # Draw the graph
    # TODO: Balance layout
    # TODO: On edge, label the argument order
    pos = nx.spring_layout(G, seed=seed)
    nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=2, arrows=True, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif", ax=ax)

    # Display the graph
    ax.margins(x=0.4)
    ax.axis("off")
    return G
