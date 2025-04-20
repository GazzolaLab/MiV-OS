__doc__ = """
Topological sorting of the graph.
"""

__all__ = ["topological_sort"]

from ..operator.protocol import _Node


def _get_upstream_topology(
    node: _Node, upstream_nodelist: list[_Node] | None = None
) -> list[_Node]:
    """
    Get the upstream topology of a node.

    Args:
        node: The node to get the upstream topology for
        upstream_nodelist: Optional list of already visited nodes

    Returns:
        List of nodes in the upstream topology
    """
    if upstream_nodelist is None:
        upstream_nodelist = []

    # Check if node is cachable
    cached_flag = False
    try:
        cached_flag = node.cacher.check_cached()
    except (AttributeError, FileNotFoundError):
        """
        For any reason when cached result could not be retrieved.

        AttributeError: Occurs when cacher is not defined
        FileNotFoundError: Occurs when cache_dir is not set or cache files doesn't exist
        """
        pass

    if not cached_flag:  # Run all upstream nodes
        for upstream_node in node.iterate_upstream():
            if upstream_node in upstream_nodelist:
                continue
            _get_upstream_topology(upstream_node, upstream_nodelist)
    upstream_nodelist.append(node)
    return upstream_nodelist


def topological_sort(node: _Node) -> list[_Node]:
    """
    Topological sort of the graph.
    Returns the list of operations in order to execute the given node.

    Args:
        node: The node to get the topological sort for

    Returns:
        List of nodes in topological order

    Raises:
        RuntimeError: If there is a loop in the graph.
    """
    upstream = _get_upstream_topology(node)

    # Track indices of nodes in the sorted list
    key = []
    pos = []
    ind = 0
    tsort: list[_Node] = []

    while len(upstream) > 0:
        key.append(upstream[-1])
        pos.append(ind)
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
