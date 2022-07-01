__all__ = ["AbnormalityDetector"]

from typing import List

from tqdm import tqdm

from miv.io import Data, DataManager
from miv.signal.filter import ButterBandpass, FilterProtocol
from miv.signal.spike import SpikeCutout, SpikeDetectionProtocol, ThresholdCutoff
from miv.typing import SpikestampsType
from miv.visualization import extract_waveforms


class AbnormalityDetector:
    def __init__(
        self,
        spontaneous_data: Data,
        experiment_data: DataManager,
        signal_filter: FilterProtocol,
        spike_detector: SpikeDetectionProtocol,
        pca_num_components: int = 3,
    ):
        self.spontaneous_data: Data = spontaneous_data
        self.data_manager: DataManager = experiment_data
        self.signal_filter: FilterProtocol = signal_filter
        self.spike_detector = spike_detector
        self.num_components: int = pca_num_components
        self.trained: bool = False

        # 1. Generate PCA cutouts for spontaneous recording
        # spontanous_cutouts = self.get_cutouts(spontaneous_data)

    def get_cutouts(self, data: Data) -> List[SpikeCutout]:
        with data.load() as (sig, times, samp):
            spontaneous_sig = self.signal_filter(sig, samp)
            spontaneous_spikes = self.spike_detector(spontaneous_sig, times, samp)

            skipped_channels = []  # Channels with not enough spikes for cutouts
            cutouts = []  # List of SpikeCutout objects
            for chan_index in tqdm(range(spontaneous_sig.shape[1])):
                try:
                    channel_cutouts = extract_waveforms(
                        spontaneous_sig, spontaneous_spikes, chan_index, samp
                    )
                    cutouts.append(SpikeCutout(channel_cutouts, samp))
                except ValueError:
                    skipped_channels.append(chan_index)
        return cutouts
