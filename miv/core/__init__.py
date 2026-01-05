# Alias/Shortcut
from ..import_helper import getter_upon_call

_submodule_paths_for_alias = {
    "pipeline": ["Pipeline"],
    "datatype.pure_python": ["PythonDataType", "NumpyDType"],
    "datatype.events": ["Events"],
    "datatype.signal": ["Signal"],
    "datatype.spikestamps": ["Spikestamps"],
    "datatype.node_mixin": ["DataNodeMixin"],
    "operator.operator": ["OperatorMixin"],
    "operator.wrapper": ["cache_call"],
    "source.wrapper": ["cached_method"],
    "operator_generator.operator": ["GeneratorOperatorMixin"],
    "operator_generator.wrapper": ["cache_generator_call"],
    "source.node_mixin": ["DataLoaderMixin"],
}
__getattr__ = getter_upon_call(__name__, _submodule_paths_for_alias)
