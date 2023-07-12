from __future__ import annotations

__doc__ = """"""
__all__ = ["_Chainable", "BaseChainingMixin"]

from typing import TypeVar  # TODO: For python 3.11, we can use typing.Self
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Union,
)

import functools
import itertools

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from miv.core.datatype import DataTypes

SelfChain = TypeVar("SelfChain", bound="_Chainable")


class _Chainable(Protocol):
    """
    Behavior includes:
        - Chaining modules in forward/backward linked lists
            - Forward direction defines execution order
            - Backward direction defines dependency order
    """

    @property
    def tag(self) -> str:
        ...

    @property
    def output(self) -> list[DataTypes]:
        ...

    def __rshift__(self, right: SelfChain) -> SelfChain:
        ...

    def iterate_upstream(self) -> Iterator[SelfChain]:
        ...

    def iterate_downstream(self) -> Iterator[SelfChain]:
        ...

    def clear_connections(self) -> None:
        ...

    def summarize(self) -> str:
        """Print summary of downstream network structures."""
        ...


class BaseChainingMixin:
    """
    Base mixin to create chaining structure between objects.

    Need further implementation of: output, tag
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline_called = False
        self._downstream_list: list[_Chainable] = []
        self._upstream_list: list[_Chainable] = []

        self._output: DataTypes | None = None

    def __rshift__(self, right: _Chainable) -> _Chainable:
        self._downstream_list.append(right)
        right._upstream_list.append(self)
        return right

    def clear_connections(self) -> None:
        """Clear all the connections to other nodes, and remove dependencies."""
        for node in self.iterate_downstream():
            node._upstream_list.remove(self)
        for node in self.iterate_upstream():
            node._downstream_list.remove(self)
        self._downstream_list.clear()
        self._upstream_list.clear()

    def iterate_downstream(self) -> Iterator[_Chainable]:
        return iter(self._downstream_list)

    def iterate_upstream(self) -> Iterator[_Chainable]:
        return iter(self._upstream_list)

    def summarize(self) -> str:  # TODO: create DFS and BFS traverser
        q = [(0, self)]
        order = []  # DFS
        while len(q) > 0:
            depth, current = q.pop(0)
            if current in order:  # Avoid loop
                continue
            q += [(depth + 1, node) for node in current.iterate_downstream()]
            order.append((depth, current))
        return self._text_visualize_hierarchy(order)

    def visualize(self, show: bool = False, seed: int = 200) -> None:
        import networkx as nx

        G = nx.DiGraph()

        # BFS
        visited = []
        next_list = [self]
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

    def _text_visualize_hierarchy(self, string_list, prefix="|__ "):
        output = []
        for i, item in enumerate(string_list):
            depth, label = item
            if depth > 0:
                output.append("    " * (depth - 1) + prefix + str(label))
            else:
                output.append(str(label))
        return "\n".join(output)

    def _get_upstream_topology(
        self, lst: list[SelfChain] | None = None
    ) -> list[SelfChain]:
        if lst is None:
            lst = []
        if (
            hasattr(self, "cacher")
            and self.cacher is not None
            and self.cacher.cache_dir is not None
            and self.cacher.check_cached()
        ):
            pass
        else:
            for node in self.iterate_upstream():
                if node in lst:
                    continue
                node._get_upstream_topology(lst)
        lst.append(self)
        return lst

    def topological_sort(self):
        """
        Topological sort of the graph.
        Raise RuntimeError if there is a loop in the graph.
        """
        # TODO: Make it free function
        upstream = self._get_upstream_topology()

        # pos = dict()  # FIXME: Operators are not hashable
        key = []
        pos = []
        ind = 0
        tsort = []

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
