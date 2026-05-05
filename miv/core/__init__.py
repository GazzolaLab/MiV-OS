# Alias/Shortcut
from ..import_helper import getter_upon_call

_submodule_paths_for_alias = {
    "pipeline": ["Pipeline"],
    "datatype.pure_python": ["PythonDataType", "NumpyDType"],
    "datatype.events": ["Events"],
    "datatype.signal": ["Signal"],
    "datatype.spikestamps": ["Spikestamps"],
    "datatype.node_mixin": [
        "DataNodeBase",  # public name
        "DataNodeMixin",  # internal name
    ],
    "operator.operator": [
        "EagerOpNodeBase",  # public name
        "OperatorMixin",  # internal name
    ],
    "operator.wrapper": ["cache_call"],  # Deprecated. Keep for backward compatibility.
    "source.wrapper": ["cached_method"],
    "operator_generator.operator": [
        "StreamOpNodeBase",  # public name
        "GeneratorOperatorMixin",  # internal name
    ],
    "operator_generator.wrapper": [
        "cache_generator_call"
    ],  # Deprecated. Keep for backward compatibility.
    "source.node_mixin": [
        "SourceNodeBase",  # public name
        "DataLoaderMixin",  # internal name
    ],
}
__getattr__ = getter_upon_call(__name__, _submodule_paths_for_alias)
