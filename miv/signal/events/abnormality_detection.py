__all__ = ["AbnormalityDetector"]

from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tqdm import tqdm

from miv.io import Data, DataManager
from miv.signal.filter import ButterBandpass, FilterProtocol
from miv.signal.spike import (
    ChannelSpikeCutout,
    PCADecomposition,
    SpikeCutout,
    SpikeDetectionProtocol,
)
from miv.typing import SpikestampsType
from miv.visualization import extract_waveforms


class AbnormalityDetector:
    """Abnormality Detector
    The initialization of this class if the first step in the process.
    With initialization, PCA cutouts for the spontaneous spikes are generated.

        Attributes
        ----------
        spontaneous_data : Data
        experiment_data : Data
        signal_filter : FilterProtocol
            Signal filter used prior to spike detection
        spike_detector : SpikeDetectionProtocol
            Spike detector used to get spiketrains from signal
        pca_num_components : int, default = 3
            The number of components in PCA decomposition
    """

    def __init__(
        self,
        spontaneous_data: Data,
        experiment_data: DataManager,
        signal_filter: FilterProtocol,
        spike_detector: SpikeDetectionProtocol,
        pca_num_components: int = 3,
        model: Optional[tf.keras.Model] = None,
    ):
        self.spontaneous_data: Data = spontaneous_data
        self.data_manager: DataManager = experiment_data
        self.signal_filter: FilterProtocol = signal_filter
        self.spike_detector = spike_detector
        self.num_components: int = pca_num_components
        self.trained: bool = False
        self.categorized: bool = False
        self.model = model if model else None

        # 1. Generate PCA cutouts for spontaneous recording
        self.spontanous_cutouts = self._get_cutouts(spontaneous_data)

    def _get_cutouts(self, data: Data) -> List[ChannelSpikeCutout]:
        pca = PCADecomposition()
        with data.load() as (sig, times, samp):
            spontaneous_sig = self.signal_filter(sig, samp)
            spontaneous_spikes = self.spike_detector(spontaneous_sig, times, samp)

            skipped_channels = []  # Channels with not enough spikes for cutouts
            exp_cutouts = []  # List of SpikeCutout objects
            for chan_index in tqdm(range(spontaneous_sig.shape[1])):
                try:
                    channel_cutouts_list: List[SpikeCutout] = []
                    raw_cutouts = extract_waveforms(
                        spontaneous_sig, spontaneous_spikes, chan_index, samp
                    )
                    labels, transformed = pca.project(self.num_components, raw_cutouts)

                    for cutout_index, raw_cutout in enumerate(raw_cutouts):
                        channel_cutouts_list.append(
                            SpikeCutout(raw_cutout, samp, labels[cutout_index])
                        )
                    exp_cutouts.append(
                        ChannelSpikeCutout(
                            channel_cutouts_list, self.num_components, chan_index
                        )
                    )
                except ValueError:
                    skipped_channels.append(chan_index)
        return exp_cutouts

    def categorize_spontaneous(
        self, categorization_list: List[List[int]]  # list[chan_index][comp_index]
    ):
        """Categorize the spontaneous recording components.
        This is the second step in the process of abnormality detection.
        This categorization provides training data for the next step.

        Parameters
        ----------
        categorization_list: List[List[int]]
            The categorization given to components of channels of the spontaneous recording.
            This is a 2D list. The row index represents the channel index. The column
            index represents the PCA component index.
        """
        for chan_index, chan_row in enumerate(categorization_list):
            self.spontanous_cutouts[chan_index].categorize(chan_row)
        self.categorized = True

    def train_model(self, layer_sizes: List[int], epochs: int = 5) -> Dict[str, any]:
        """Create and train model for cutout recognition
        This is the third step in the process of abnormality detection.

        Parameters
        ----------
        layer_sizes : List[int]
            The number of nodes in each layer.
            For example, a first layer with 256 nodes and a second with 64 nodes
            would be [256, 64].
        epochs : int, default = 5
            The number of iterations for model training

        Returns
        -------
        test_loss : float
        test_accuracy : float
        """
        # Get the labeled cutouts
        labeled_cutouts = []
        labels = []
        size = 0
        for chan_index, channelCutout in enumerate(self.spontanous_cutouts):
            channel_labeled_cutouts = channelCutout.get_labeled_cutouts()
            labeled_cutouts.append(channel_labeled_cutouts["labeled_cutouts"])
            labels.append(channel_labeled_cutouts["labels"])
            size += channel_labeled_cutouts["size"]

        # Shuffle the cutouts and split into training and test portions
        labeled_cutouts, labels = shuffle(labeled_cutouts, labels)
        split = int(size * 0.8)
        train_cutouts = labeled_cutouts[:split]
        train_labels = labels[:split]
        test_cutouts = labeled_cutouts[split:]
        test_labels = labels[:split]

        # Set up and train model
        layers = [tf.keras.layers.Dense(90)]
        for layer_size in layer_sizes:
            layers.append(tf.keras.layers.Dense(layer_size))
        layers.append(
            tf.keras.layers.Dense(len(self.spontanous_cutouts[0].CATEGORY_NAMES[1:]))
        )
        model = tf.keras.Sequential(layers)
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        model.fit(train_cutouts, train_labels, epochs=epochs)

        # Test and return model
        test_loss, test_acc = model.evaluate(test_cutouts, test_labels)
        self.model = model
        self.trained = True
        return {"test_loss": test_loss, "test_accuracy": test_acc}
