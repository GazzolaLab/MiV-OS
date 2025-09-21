# Alias/Shortcut
from ..import_helper import getter_upon_call

_submodule_paths_for_alias = {
    # File I/O
    "file.read": [
        "read",
        "select_datasets",
        "validate_subset",
        "calculate_index_from_counters",
        "unpack",
        "get_ncontainers_in_file",
        "get_ncontainers_in_data",
        "get_file_metadata",
        "print_file_metadata",
    ],
    "file.write": [
        "initialize",
        "clear_container",
        "create_container",
        "create_group",
        "create_dataset",
        "pack",
        "convert_list_and_key_to_string_data",
        "convert_dict_to_string_data",
        "write_metadata",
        "write",
    ],
    "file.import_signal": ["ImportSignal"],
    # OpenEphys
    "openephys.data": ["Data", "DataManager"],
    "openephys.binary": [
        "apply_channel_mask",
        "bits_to_voltage",
        "oebin_read",
        "load_ttl_event",
        "load_recording",
        "load_continuous_data",
        "load_timestamps",
    ],
    # Intan
    "intan.data": ["DataIntan", "DataIntanTriggered"],
    "intan.rhs": [
        "plural",
        "read_qstring",
        "read_header",
        "notch_filter",
        "find_channel_in_group",
        "find_channel_in_header",
        "get_bytes_per_data_block",
        "read_one_data_block",
        "data_to_result",
        "plot_channel",
        "load_file",
        "print_all_channel_names",
        "print_names_in_group",
    ],
    # Serial communication
    "serial.arduino": ["ArduinoSerial", "list_serial_ports"],
    "serial.stimjim": ["StimjimSerial"],
    # ASDF
    "asdf.asdf": ["DataASDF"],
    # Simulator
    "simulator.data": ["Data"],
    # Protocol
    "protocol": ["DataProtocol"],
}
__getattr__ = getter_upon_call(__name__, _submodule_paths_for_alias)
