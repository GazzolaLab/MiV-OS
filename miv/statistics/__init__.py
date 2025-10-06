from typing import no_type_check

# Alias/Shortcut
from ..import_helper import getter_upon_call

_submodule_paths_for_alias = {
    "baks": ["bayesian_adaptive_kernel_smoother"],
    "burst": ["burst_detection", "burst_array"],
    "info_theory": [
        "probability_distribution",
        "shannon_entropy",
        "block_entropy",
        "entropy_rate",
        "active_information",
        "mutual_information",
        "relative_entropy",
        "joint_entropy",
        "conditional_entropy",
        "cross_entropy",
        "transfer_entropy",
        "partial_information_decomposition",
    ],
    "causality": ["pairwise_causality"],
    "peristimulus_analysis": [
        "PSTH",
        "PeristimulusActivity",
        "peri_stimulus_time",
        "PSTHOverlay",
    ],
    "signal_statistics": ["signal_to_noise", "spike_amplitude_to_background_noise"],
    "spiketrain_statistics": [
        "firing_rates",
        "MFRComparison",
        "interspike_intervals",
        "coefficient_variation",
        "binned_spiketrain",
        "fano_factor",
        "spike_counts_with_kernel",
        "decay_spike_counts",
        "instantaneous_spike_rate",
    ],
}
__getattr__ = getter_upon_call(__name__, _submodule_paths_for_alias)
