from __future__ import annotations

__doc__ = """"""
__all__ = ["BaseChainingMixin"]

from typing import TYPE_CHECKING, Any
from collections.abc import Iterator

import itertools

import networkx as nx
import matplotlib.pyplot as plt


if TYPE_CHECKING:
    from .protocol import _Chainable


class BaseChainingMixin:
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

    def visualize(self, show: bool = False, seed: int = 200) -> nx.DiGraph:
        """
        Visualize the network structure of the "Operator".
        """
        G = nx.DiGraph()

        # BFS
        visited: list[_Chainable] = []
        next_list: list[_Chainable] = [self]
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
        nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=2, arrows=True)
        nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

        # Display the graph
        plt.margins(x=0.4)
        plt.axis("off")
        if show:
            plt.show()
        return G

    def _text_visualize_hierarchy(
        self,
        string_list: list[tuple[int, _Chainable]],
        prefix: str = "|__ ",
    ) -> str:
        output = []
        for _i, item in enumerate(string_list):
            depth, label = item
            if depth > 0:
                output.append("    " * (depth - 1) + prefix + str(label))
            else:
                output.append(str(label))
        return "\n".join(output)
