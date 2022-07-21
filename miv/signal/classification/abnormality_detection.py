__all__ = ["AbnormalityDetector"]

from typing import Any, Dict, List, Optional

import numpy as np
import quantities as pq
import tensorflow as tf
from neo import SpikeTrain
from sklearn.utils import shuffle
from tqdm import tqdm

from miv.io import Data, DataManager
from miv.signal.classification.neuronal_spike_classification import (
    NeuronalSpikeClassifier,
)
from miv.signal.classification.protocol import SpikeClassificationModelProtocol
from miv.signal.filter import ButterBandpass, FilterProtocol
from miv.signal.spike import (
    ChannelSpikeCutout,
    SpikeCutout,
    SpikeDetectionProtocol,
    SpikeFeatureExtractionProtocol,
)
from miv.typing import SpikestampsType
from miv.visualization import extract_waveforms


class AbnormalityDetector:
    """Abnormality Detector

    This class integrates NeuronalSpikeClassifier with Data and SpikeFeatureExtractionProtocol


        Attributes
        ----------
        spontaneous_data : Data
        experiment_data : Data
        spont_signal_filter : FilterProtocol
            Spontaneous signal filter used prior to spike detection
        spont_spike_detector : SpikeDetectionProtocol
            Spontaneous spike detector used to get spiketrains from signal
        spike_feature_extractor : SpikeFeatureExtractionProtocol
            Spike feature extraction object (PCA, Wavelet, etc.) used to group
            the spontaneous cutouts
        extractor_decomposition_parameter : int, default = 3
    """

    def __init__(
        self,
        spontaneous_data: Data,
        spontaneous_signal_filter: FilterProtocol,
        spontaneous_spike_detector: SpikeDetectionProtocol,
        spike_feature_extractor: SpikeFeatureExtractionProtocol,
        extractor_decomposition_parameter: int = 3,
    ) -> None:

        self.spontaneous_cutouts = self._get_all_cutouts(
            spontaneous_data,
            spontaneous_signal_filter,
            spontaneous_spike_detector,
            spike_feature_extractor,
            extractor_decomposition_parameter,
        )
        self.classifier: NeuronalSpikeClassifier
        self.categorized: bool = False

    def _get_all_cutouts(
        self,
        data: Data,
        signal_filter: FilterProtocol,
        spike_detector: SpikeDetectionProtocol,
        extractor: SpikeFeatureExtractionProtocol,
        extractor_decomp_param: int,
    ) -> np.ndarray:
        with data.load() as (sig, times, samp):
            filtered_sig = signal_filter(sig, samp)
            spikestamps = spike_detector(filtered_sig, times, samp)

            num_channels = np.shape(filtered_sig)[1]
            return_cutouts = np.ndarray(num_channels)

            for chan_index in tqdm(range(num_channels)):
                channel_spikes = spikestamps[chan_index]

                if np.size(channel_spikes) < extractor_decomp_param:
                    return_cutouts[chan_index] = ChannelSpikeCutout(
                        np.array([]), extractor_decomp_param, chan_index
                    )

                else:
                    raw_cutouts: np.ndarray = extract_waveforms(
                        filtered_sig, spikestamps, chan_index, samp
                    )

                    labels, transformed = extractor.project(
                        extractor_decomp_param, raw_cutouts
                    )

                    channel_cutouts_list = []
                    for cutout_index, raw_cutout in enumerate(raw_cutouts):
                        channel_cutouts_list.append(
                            SpikeCutout(
                                raw_cutout,
                                samp,
                                extractor_decomp_param,
                                spikestamps[chan_index][cutout_index],
                            )
                        )

                    return_cutouts[chan_index] = ChannelSpikeCutout(
                        np.array(
                            channel_cutouts_list,
                        ),
                        extractor_decomp_param,
                        chan_index,
                    )
        return return_cutouts

    def categorize_spontaneous(
        self,
        categorization_list: np.ndarray,  # categorization_list[chan_index][comp_index]
    ) -> None:
        """Categorize the spontaneous recording components.
        This is the second step in the process of abnormality detection.
        This categorization provides training data for the next step.

        Parameters
        ----------
        categorization_list : np.ndarray
            The categorization given to components of channels of the spontaneous recording.
            This is a 2D Numpy array. The row index represents the channel index. The column
            index represents the extractor component index.
        """
        for chan_index, chan_row in enumerate(categorization_list):
            self.spontaneous_cutouts[chan_index].categorize(chan_row)
        self.categorized = True

    def _get_all_labeled_cutouts(self, shuffle_cutouts: bool = True) -> Dict[str, Any]:
        labeled_cutouts = []
        labels = []
        size = 0
        for chan_index, channel_spike_cutout in enumerate(self.spontaneous_cutouts):
            channel_labeled_cutouts = channel_spike_cutout.get_labeled_cutouts()

            for cutout_index, cutout_label in enumerate(
                channel_labeled_cutouts["labels"]
            ):
                labels.append(cutout_label)
                labeled_cutouts.append(
                    np.array(channel_labeled_cutouts["cutouts"][cutout_index])
                )
                size += 1

        if shuffle_cutouts:
            labeled_cutouts, labels = shuffle(labeled_cutouts, labels)

        return {
            "labels": np.array(labels),
            "cutouts": np.array(labeled_cutouts),
            "size": size,
        }

    def init_classifier(
        self,
        model: Optional[SpikeClassificationModelProtocol] = None,
    ) -> None:
        """Initialize Neuronal Spike Classifier

        Uses the categorized cutouts to initiate a NeuronalSpikeClassifier object

        Parameters
        ----------
        model : Optional[SpikeClassificationModelProtocol], default = None
            The model used by the classfier.
            By default, a standard tensorflow keras model will be created
        """
        self.classifier = NeuronalSpikeClassifier(model)

    def train_classifier_model(
        self, train_test_split: float, **model_fit_kwargs
    ) -> None:
        """Train classifier model with labeled spontaneous cutouts

        train_test_split : float
            The proportion of labeled spontaneous cutouts used for training
            For example, if there are 10 labeled cutouts and 8 is used for
            training, then train_test_split would be 0.8.
        """

        all_labeled_cutouts = self._get_all_labeled_cutouts(shuffle=True)
        split_index = int(train_test_split * all_labeled_cutouts["size"])
        train_labels = all_labeled_cutouts["labels"][:split_index]
        train_cutouts = all_labeled_cutouts["cutouts"][:split_index]

        self.classifier.train_model(train_cutouts, train_labels, model_fit_kwargs)

    def evaluate_model(
        self,
        test_cutouts: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Evaluate the trained model

        Parameters
        ----------
        test_cutouts : Optional[np.ndarray], default = None
            Raw 2D numpy array of rows of cutout sample points
        test_labels : Optional[np.ndarray], default = None
            1D numpy array of categorization indices

        Returns
        -------
        test_loss : float
            1 if model was not trained
        test_accuracy : float
            0 if model was not trained
        test_precision : float
        test_recall : float
        test_f1 : float
        """

        if not self.trained:
            return {"test_loss": 1, "test_accuracy": 0}

        data = test_cutouts if test_cutouts is not None else self.test_cutouts
        labels = test_labels if test_labels is not None else self.test_labels

        loss, acc = self.model.evaluate(data, labels)

        return {"test_loss": loss, "test_accuracy": acc}

    def get_only_neuronal_spikes(
        self,
        exp_data: Data,
        accuracy_threshold: float = 0.9,
        signal_filter: Optional[FilterProtocol] = None,
        spike_detector: Optional[SpikeDetectionProtocol] = None,
    ) -> List[SpikestampsType]:
        """This method takes each signle individual spike in the experiment data
        and uses the trained model to predict whether the spike is neuronal.

        Parameters
        ----------
        exp_data : Data
            Experiment data
        accuracy_threshold : float, defualt = 0.9
            The category prediction comes in probabilities for each category.
            If the probability for "neuronal" is higher than this threshold,
            the spike will be included.
        signal_filter : Optional[FilterProtocol], default = None
            Filter applied to the experiment data before spike detection.
            If left empty or None, the same filter for the spontaneous data
            will be used.
        spike_detector : Optional[SpikeDetectionProtocol], default = None
            The spike detector used to detect spikes on the experiment data.
            If left empty or None, the same spike detector for the spontaneous
            data will be used.

        Returns
        -------
        (Same return format as SpikeDetectionProtocol)
        A list of SpikestampsType for each channel
        """
        if not self.trained:
            raise Exception("Abnormality detector is not trained yet.")

        exp_filter = signal_filter if signal_filter else self.spont_signal_filter
        exp_detector = spike_detector if spike_detector else self.spont_spike_detector
        list_of_cutout_channels = self._get_cutouts(
            exp_data, exp_filter, exp_detector, self.extractor
        )
        new_spiketrains: List[SpikestampsType] = []

        prob_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        for chan_index, cutout_channel in tqdm(enumerate(list_of_cutout_channels)):
            times = []
            t_stop = 0
            for cutout_index, spike_cutout in enumerate(cutout_channel.cutouts):
                prediction = prob_model.predict(
                    np.array([spike_cutout.cutout]), verbose=0
                )
                if prediction[0][0] >= accuracy_threshold:
                    times.append(spike_cutout.time)
                    t_stop = spike_cutout.time
            new_spiketrains.append(SpikeTrain(times, t_stop, pq.s))

        return new_spiketrains

    def get_only_neuronal_components(
        self,
        exp_data: Data,
        accuracy_threshold: float = 0.9,
        signal_filter: Optional[FilterProtocol] = None,
        spike_detector: Optional[SpikeDetectionProtocol] = None,
    ) -> List[SpikestampsType]:
        """This method first takes each spike in a single component group and
        computer an average spike. Then, this average spike is used to determine
        whether the component is neuronal and should be returned.

        Parameters
        ----------
        exp_data : Data
            Experiment data
        accuracy_threshold : float, defualt = 0.9
            The category prediction comes in probabilities for each category.
            If the probability for "neuronal" is higher than this threshold,
            the spikes in the component will be included.
        signal_filter : Optional[FilterProtocol], default = None
            Filter applied to the experiment data before spike detection.
            If left empty or None, the same filter for the spontaneous data
            will be used.
        spike_detector : Optional[SpikeDetectionProtocol], default = None
            The spike detector used to detect spikes on the experiment data.
            If left empty or None, the same spike detector for the spontaneous
            data will be used.

        Returns
        -------
        (Same return format as SpikeDetectionProtocol)
        A list of SpikestampsType for each channel
        """
        if not self.trained:
            raise Exception("Abnormality detector is not trained yet.")

        exp_filter = signal_filter if signal_filter else self.spont_signal_filter
        exp_detector = spike_detector if spike_detector else self.spont_spike_detector
        list_of_cutout_channels = self._get_cutouts(
            exp_data, exp_filter, exp_detector, self.extractor
        )
        new_spiketrains: List[SpikestampsType] = []
        prob_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

        for chan_index, cutout_channel in tqdm(enumerate(list_of_cutout_channels)):
            chan_cutouts_by_comp = cutout_channel.get_cutouts_by_component()
            channel_times = []

            for comp_index, comp_cutouts in enumerate(chan_cutouts_by_comp):
                loaded_comp_cutouts: List[List[float]] = []

                for cutout_index, spike_cutout in comp_cutouts:
                    loaded_comp_cutouts.append(spike_cutout.cutout)

                comp_mean_cutout = np.mean(loaded_comp_cutouts, axis=0)
                prediction = prob_model.predict(comp_mean_cutout, verbose=0)

                if prediction[0] >= accuracy_threshold:
                    for cutout_index, spike_cutout in comp_cutouts:
                        channel_times.append(spike_cutout.time)

            t_stop = 0 if not (channel_times) else channel_times[-1]
            new_spiketrains.append(SpikeTrain(channel_times, t_stop, pq.s))

        return new_spiketrains
