# Alias/Shortcut
from ..import_helper import getter_upon_call

_submodule_paths_for_alias = {
    # Filter module
    "filter.butter_bandpass_filter": ["ButterBandpass"],
    "filter.median_filter": ["MedianFilter"],
    "filter.notch_filter": ["Notch"],
    # Spike module
    "spike.detection": [
        "ThresholdCutoff",
        "ThresholdCutoffNonSparse",
        # "query_firing_rate_between",  # TODO: Maybe make this into operator
    ],
    "spike.sorting": [
        "SpikeSorting",
        "SuperParamagneticClustering",
        "PCADecomposition",
        "PCAClustering",
        "WaveletDecomposition",
    ],
    "spike.waveform": ["ExtractWaveforms", "WaveformAverage"],
    "spike.waveform_alignment": ["WaveformAlignment"],
    "spike.waveform_statistical_filter": ["WaveformStatisticalFilter"],
    # Similarity module
    "similarity.simple": [
        "domain_distance_matrix",
    ],
    "similarity.dtw": ["dynamic_time_warping_distance_matrix"],
    # Core signal utilities
    "downsampling": ["Downsample"],
}
__getattr__ = getter_upon_call(__name__, _submodule_paths_for_alias)
