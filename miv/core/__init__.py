from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .datatype.spikestamps import Spikestamps

# Alias/Shortcut
from ..import_helper import getter_upon_call

_submodule_paths_for_alias = {
    "pipeline": ["Pipeline"],
    "datatype.events": ["Events"],
    "datatype.signal": ["Signal"],
    "datatype.spikestamps": ["Spikestamps"],
    "operator.operator": ["DataLoaderMixin", "DataNodeMixin", "OperatorMixin"],
    "operator.wrapper": ["cache_call", "cached_method"],
    "operator_generator.operator": ["GeneratorOperatorMixin"],
    "operator_generator.wrapper": ["cache_generator_call"],
}
__getattr__ = getter_upon_call(__name__, _submodule_paths_for_alias)
