__doc__ = """"""
__all__ = ["_Chainable", "BaseChainingMixin"]

from typing import TypeVar  # TODO: For python 3.11, we can use typing.Self
from typing import Callable, Iterator, List, Optional, Protocol, Set, Union

import functools
import itertools

import matplotlib.pyplot as plt

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
    def output(self) -> List[DataTypes]:
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
    """

    def __init__(self):
        super().__init__()
        self._downstream_list: List[_Chainable] = []
        self._upstream_list: List[_Chainable] = []

        self._output: Optional[DataTypes] = None

    def __rshift__(self, right: _Chainable) -> _Chainable:
        self._downstream_list.append(right)
        right._upstream_list.append(self)
        return right

    def clear_connections(self) -> None:
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

    def visualize(self, show: bool = False) -> None:
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
        pos = nx.spring_layout(G, seed=200)
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

    def _get_full_topology(self) -> List[SelfChain]:
        """Get all the operators and data loaders in the topology."""
        visited = []
        next_list = [self]
        while next_list:
            v = next_list.pop()
            visited.append(v)
            for node in itertools.chain(v.iterate_downstream(), v.iterate_upstream()):
                if node in visited or node in next_list:
                    continue
                next_list.append(node)
        return visited

    def topological_sort(self):
        """Topological sort of the topology."""
        # TODO: Make it free function
        # TODO: Find loop detection
        # Mark all the vertices as not visited
        all_nodes = self._get_full_topology()
        visited = []

        stack = []

        # Call the helper function to store Topological
        # Sort starting from all vertices one by one
        for v, seed_node in enumerate(all_nodes):
            if seed_node in visited:
                continue
            # working stack contains key and the corresponding current generator
            working_stack = [seed_node]
            while working_stack:
                # get last element from stack
                node = working_stack.pop()
                visited.append(node)

                # run through neighbor generator until it's empty
                for next_node in node.iterate_downstream():
                    if next_node in visited:
                        continue
                    # remember current work
                    working_stack.append(node)  # TODO: Somebody check this line
                    # restart with new neighbor
                    working_stack.append(next_node)
                    break
                else:
                    # no already-visited neighbor (or no more of them)
                    stack.append(node)

        # Print contents of the stack in reverse
        stack.reverse()
        return stack


def test_topological_sort():
    class V(BaseChainingMixin):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def __repr__(self):
            return str(self.name)

    a = V(1)
    b = V(2)
    c = V(3)
    d = V(4)
    e = V(5)

    a >> b >> c >> d >> e
    print(a.summarize())
    print(b.summarize())
    print(a.topological_sort())
    print(b.topological_sort())

    a.clear_connections()
    print(a.summarize())
    print(b.summarize())
    print(a.topological_sort())
    print(b.topological_sort())

    a.clear_connections()
    b.clear_connections()
    c.clear_connections()
    d.clear_connections()
    e.clear_connections()

    a >> e
    b >> c >> d >> e
    print(a.summarize())
    print(b.summarize())
    print(a.topological_sort())
    print(b.topological_sort())


if __name__ == "__main__":
    test_topological_sort()
