from typing import no_type_check

# Alias/Shortcut
from ...import_helper import getter_upon_call

_submodule_paths_for_alias = {
    "centrality": ["plot_eigenvector_centrality"],
    "connectivity": ["DirectedConnectivity", "UndirectedConnectivity"],
    "instantaneous_connectivity": ["InstantaneousConnectivity"],
}
__getattr__ = getter_upon_call(__name__, _submodule_paths_for_alias)
