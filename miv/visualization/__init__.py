# Alias/Shortcut
from ..import_helper import getter_upon_call

_submodule_paths_for_alias = {
    "causality": ["pairwise_causality_plot", "spike_triggered_average_plot"],
    "replay": ["ReplayRecording"],
    "event": ["plot_spiketrain_raster", "plot_burst"],
    "raw_signal": ["MultiChannelSignalVisualization"],
    "activity": ["NeuralActivity"],
    "connectivity": ["plot_connectivity", "plot_connectivity_interactive"],
    "fft_domain": ["plot_frequency_domain", "plot_spectral"],
}
__getattr__ = getter_upon_call(__name__, _submodule_paths_for_alias)
