from __future__ import annotations

__doc__ = """"""
__all__ = ["BaseChainingMixin"]

from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Set, Union, cast
from collections.abc import Callable, Iterator
from typing_extensions import Self

import functools
import itertools

import networkx as nx
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from miv.core.datatype import DataTypes
    from .protocol import _Chainable, OperatorNode


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
        self._downstream_list.append(right)
        right._upstream_list.append(cast("_Chainable", self))
        return right

    def clear_connections(self) -> None:
        """Clear all the connections to other nodes, and remove dependencies."""
        for node in self.iterate_downstream():
            node._upstream_list.remove(cast("_Chainable", self))
        for node in self.iterate_upstream():
            node._downstream_list.remove(cast("_Chainable", self))
        self._downstream_list.clear()
        self._upstream_list.clear()

    def iterate_downstream(self) -> Iterator[_Chainable]:
        return iter(self._downstream_list)

    def iterate_upstream(self) -> Iterator[_Chainable]:
        return iter(self._upstream_list)

    def summarize(self) -> str:  # TODO: create DFS and BFS traverser
        q: list[tuple[int, _Chainable]] = [(0, cast("_Chainable", self))]
        order: list[tuple[int, _Chainable]] = []  # DFS
        while len(q) > 0:
            if q[0] in order:  # Avoid loop
                continue
            depth, current = q.pop(0)
            q += [(depth + 1, node) for node in current.iterate_downstream()]
            order.append((depth, current))
        return self._text_visualize_hierarchy(order)

    def visualize(self: _Chainable, show: bool = False, seed: int = 200) -> nx.DiGraph:
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
                G.add_edge(v.tag, node.tag)
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
        for i, item in enumerate(string_list):
            depth, label = item
            if depth > 0:
                output.append("    " * (depth - 1) + prefix + str(label))
            else:
                output.append(str(label))
        return "\n".join(output)

    def _get_upstream_topology(
        self, upstream_nodelist: list[_Chainable] | None = None
    ) -> list[_Chainable]:
        if upstream_nodelist is None:
            upstream_nodelist = []

        # Optional for Operator to be 'cachable'
        try:
            _self = cast("_Chainable", self)
            cached_flag = _self.cacher.check_cached()
        except (AttributeError, FileNotFoundError):
            """
            For any reason when cached result could not be retrieved.

            AttributeError: Occurs when cacher is not defined
            FileNotFoundError: Occurs when cache_dir is not set or cache files doesn't exist
            """
            cached_flag = False

        if not cached_flag:  # Run all upstream nodes
            for node in self.iterate_upstream():
                if node in upstream_nodelist:
                    continue
                node._get_upstream_topology(upstream_nodelist)
        upstream_nodelist.append(cast("_Chainable", self))
        return upstream_nodelist

    def topological_sort(self) -> list[_Chainable]:
        """
        Topological sort of the graph.
        Returns the list of operations in order to execute "self"

        Raise RuntimeError if there is a loop in the graph.
        """
        # TODO: Make it free function
        upstream = self._get_upstream_topology()

        # pos = dict()  # FIXME: Operators are not hashable
        key = []
        pos = []
        ind = 0
        tsort: list[_Chainable] = []

        while len(upstream) > 0:
            key.append(upstream[-1])
            pos.append(ind)
            # pos[upstream[-1]] = ind
            tsort.append(upstream[-1])
            ind += 1
            upstream.pop()
        for source in tsort:
            for up in source.iterate_upstream():
                if up not in key:
                    continue
                before = pos[key.index(source)]
                after = pos[key.index(up)]

                # If parent vertex does not appear first
                if before > after:
                    raise RuntimeError(
                        f"Found loop in operation stream: node {source} is already in the upstream : {up}."
                    )

        tsort.reverse()
        return tsort
