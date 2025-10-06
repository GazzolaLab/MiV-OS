# Alias/Shortcut
from ...import_helper import getter_upon_call

_submodule_paths_for_alias = {
    "avalanche_analysis": ["AvalancheDetection", "AvalancheAnalysis"],
}
__getattr__ = getter_upon_call(__name__, _submodule_paths_for_alias)
