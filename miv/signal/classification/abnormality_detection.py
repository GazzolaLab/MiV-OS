__all__ = ["DetectorWithSpontaneousData"]

from typing import Any, Dict, List, Optional, Tuple, Union

import os

import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

from miv.io import Data
from miv.signal.classification.neuronal_spike_classification import (
    NeuronalSpikeClassifier,
)
from miv.signal.classification.protocol import SpikeClassificationModelProtocol
from miv.signal.filter import FilterProtocol
from miv.signal.spike import (
    ChannelSpikeCutout,
    SpikeCutout,
    SpikeDetectionProtocol,
    SpikeFeatureExtractionProtocol,
)
from miv.visualization import extract_waveforms


class DetectorWithTrainData:
    """Abnormality Detector that works by using train data file with labeled spikes

    Train data files may be huge. This class integrates the file with classifier.

        Attributes
        ----------
        train_datapath : str
        classifier : NeuronalSpikeClassifier


    """

    def __init__(self, train_datapath: str) -> None:
        if not os.path.exists(train_datapath):
            raise Exception("train_datapath does not exist!")
        self.train_datapath: str = train_datapath
        self.classifier: NeuronalSpikeClassifier

    def init_classifier(
        self, model: Optional[SpikeClassificationModelProtocol] = None
    ) -> None:
        """Initiate NeuronalSpikeClassifier object

        Parameters
        ----------
        model : Optional[SpikeClassificationModelProtocol], default = None
            Classification model used to initiate classifier.
            If left as None, a default TF Keras model will be built.
        """
        self.classifier = NeuronalSpikeClassifier(model)

        if model is None:
            with np.load(self.train_datapath) as file:
                spikes = file["spike"]
                spike_length = len(spikes[0])
                del spikes
            self.classifier.create_default_tf_keras_model(spike_length)

    def default_compile_and_train(self) -> None:
        """Compile and train classifier model with train data"""

    def get_only_neuronal_spikes(self, exp_data: Data) -> np.ndarray:
        """Get spiketrains where only neuronal spikes are present

        This function uses the classifier to select spikes.

        Parameters
        ----------
        exp_data : Data
        """

        return np.ndarray([])


class DetectorWithSpontaneousData:
    """Abnormality Detector that works by using categorized spontaneous recording

    This class integrates NeuronalSpikeClassifier with Data and SpikeFeatureExtractionProtocol

    This class is shown to be somewhat useless as the extracted spikes may not be similar within
    each component, making categorization impossible.


        Attributes
        ----------
        spontaneous_data : Data
        spontaneous_cutouts : np.ndarray
        classifier : NeuronalSpikeClassifier
        categorized : bool
            Note: this value only marks whether categorize_spontaneous() is called with
            a non-empty categorization array. Each ChannelSpikeCutout in spontaenous_cutouts
            also has a categorized value for that specific channel.
        test_labels : np.ndarray
        test_cutouts : np.ndarray
    """

    def __init__(
        self,
        spontaneous_data: Data,
        spontaneous_signal_filter: FilterProtocol,
        spontaneous_spike_detector: SpikeDetectionProtocol,
        spike_feature_extractor: SpikeFeatureExtractionProtocol,
        extractor_decomposition_parameter: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        spontaneous_data : Data
        spontaneous_signal_filter : FilterProtocol
        spontaneous_spike_detector : SpikeDetectionProtocol
        spike_feature_extractor : SpikeFeatureExtractorProtocol
        extractor_decomposition_parameter : int, default = 3

        """
        self.spontaneous_data: Data = spontaneous_data
        self.spontaneous_cutouts: np.ndarray = self._get_all_cutouts(
            spontaneous_data,
            spontaneous_signal_filter,
            spontaneous_spike_detector,
            spike_feature_extractor,
            extractor_decomposition_parameter,
        )
        self.classifier: NeuronalSpikeClassifier
        self.categorized: bool = False
        self.test_labels: np.ndarray
        self.test_cutouts: np.ndarray
        self.extractor_decomposition_parameter = extractor_decomposition_parameter

    def _check_categorized(self):
        if not self.categorized:
            raise Exception(
                "No cutouts are labeled yet. Try categorize_spontaneous method"
            )

    def _check_classifier_initiated(self):
        if not self.classifier:
            raise Exception("No classifier is initiated yet. Try init_classifier()")

    def _get_all_cutouts(
        self,
        data: Data,
        signal_filter: FilterProtocol,
        spike_detector: SpikeDetectionProtocol,
        extractor: SpikeFeatureExtractionProtocol,
        extractor_decomp_param: int,
    ) -> np.ndarray:
        """Get all cutouts from data

        Parameters
        ----------
        data : Data
        signal_filter : FilterProtocol
        spike_detector : SpikeDetectionProtocol
        extractor : SpikeFeatureExtractionProtocol
        extractor_decomp_param : int

        Returns
        -------
        Numpy array of rows of ChannelSpikeCutout objects
        """
        with data.load() as (sig, times, samp):
            filtered_sig = signal_filter(sig, samp)
            spikestamps = spike_detector(filtered_sig, times, samp)

            num_channels = np.shape(filtered_sig)[1]
            return_cutouts: np.ndarray = np.ndarray(
                num_channels, dtype=ChannelSpikeCutout
            )

            for chan_index in tqdm(range(num_channels)):
                channel_spikes = spikestamps[chan_index]

                # Not enough spikes, ignore channel
                if np.size(channel_spikes) < extractor_decomp_param:
                    return_cutouts[chan_index] = ChannelSpikeCutout(
                        np.array([]), extractor_decomp_param, chan_index
                    )

                else:
                    raw_cutouts: Union[
                        np.ndarray, Tuple[np.ndarray, np.ndarray]
                    ] = extract_waveforms(filtered_sig, spikestamps, chan_index, samp)

                    labels, transformed = extractor.project(
                        extractor_decomp_param, raw_cutouts
                    )

                    channel_cutouts_list = []
                    for cutout_index, raw_cutout in enumerate(raw_cutouts):
                        channel_cutouts_list.append(
                            SpikeCutout(
                                raw_cutout,
                                samp,
                                labels[cutout_index],
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

    def categorize_spontaneous(self, categorization_list: np.ndarray) -> None:
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
        decomp_param = self.spontaneous_cutouts[0].num_components
        if np.shape(categorization_list)[1] != decomp_param:
            raise IndexError(
                "Number of category indices does not match the extractor decomposition parameter."
            )
        num_channels = np.shape(self.spontaneous_cutouts)[0]
        if np.shape(categorization_list)[0] != num_channels:
            raise IndexError(
                "Number of channels in categorization list does not match the number of channels from spontaneous cutouts."
            )

        for chan_index, chan_row in enumerate(categorization_list):
            self.categorized = True
            self.spontaneous_cutouts[chan_index].categorize(chan_row)

    def _get_all_labeled_cutouts(self, shuffle_cutouts: bool = True) -> Dict[str, Any]:
        """Get all cutouts that are labeled

        Parameters
        ----------
        shuffle_cutouts : bool, default = True
            Shuffle cutouts before returning. This is recommended if these labels and cutouts
            will be used for model training.

        Returns
        -------
        labels :
            1D Numpy array of labels
        cutouts :
            2D Numpy array of cutouts with each row being a cutout's data points
        sizes :
            1D Numpy array for number of labeled cutouts for each channel
        """
        self._check_categorized()

        labeled_cutouts = []
        labels = []
        sizes: np.ndarray = np.ndarray(len(self.spontaneous_cutouts))
        for chan_index, channel_spike_cutout in enumerate(self.spontaneous_cutouts):
            channel_labeled_cutouts = channel_spike_cutout.get_labeled_cutouts()

            for cutout_index, cutout_label in enumerate(
                channel_labeled_cutouts["labels"]
            ):
                labels.append(cutout_label)
                labeled_cutouts.append(
                    np.array(channel_labeled_cutouts["cutouts"][cutout_index])
                )
            sizes[chan_index] = channel_labeled_cutouts["size"]

        if shuffle_cutouts:
            labeled_cutouts, labels = shuffle(labeled_cutouts, labels)

        return {
            "labels": np.array(labels),
            "cutouts": np.array(labeled_cutouts),
            "sizes": sizes,
        }

    def get_spontaneous_cutouts_by_components(self) -> np.ndarray:
        """Get all spontaneous cutouts in a 2D array
        Note: The raw spike signals are returned, instead of SpikeCutout objects.

        Returns
        -------
        3D Numpy array [component index][spike index][sampling point in raw spike]
        """

        result = [[] for i in range(self.extractor_decomposition_parameter)]

        for chan_index, chan_spike_cutout in enumerate(self.spontaneous_cutouts):
            chan_cutouts_by_comp = chan_spike_cutout.get_cutouts_by_component()

            for comp_index, comp_spikes in enumerate(chan_cutouts_by_comp):
                for spike_index, spike in enumerate(comp_spikes):
                    result[comp_index].append(spike)

        return np.array(result, dtype=object)

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
            If left as None, a standard tensorflow keras model will be created
        """
        self._check_categorized()
        self.classifier = NeuronalSpikeClassifier(model)

    def compile_classifier_model(self, **model_compile_kwargs) -> None:
        """Compile classifier model

        Parameters
        ----------
        **model_compile_kwargs
            keyworded arguments for model.compile()
        """
        self._check_classifier_initiated()
        self.classifier.compile_model(**model_compile_kwargs)

    def default_init_and_compile_classifier(self) -> None:
        """Initialize and compile classifier with default settings"""
        self.init_classifier()
        input_size = len(self.spontaneous_cutouts[0].cutouts[0])
        self.classifier.create_default_tf_keras_model(input_size)
        self.classifier.default_compile_model()

    def train_classifier_model(
        self, train_test_split: float, **model_fit_kwargs
    ) -> None:
        """Train classifier model with labeled spontaneous cutouts

        train_test_split : float
            The proportion of labeled spontaneous cutouts used for training
            For example, if there are 10 labeled cutouts and 8 is used for
            training, then train_test_split would be 0.8.
        **model_fit_kwargs
            Keyworded arguments for model.fit()
            Note: x=train_cutouts and y=train_labels are already included and should
            not be written again in this value.
        """
        self._check_classifier_initiated()
        self._check_categorized()

        all_labeled_cutouts = self._get_all_labeled_cutouts(shuffle_cutouts=True)
        split_index = int(train_test_split * len(all_labeled_cutouts["labels"]))
        train_labels = all_labeled_cutouts["labels"][:split_index]
        train_cutouts = all_labeled_cutouts["cutouts"][:split_index]

        self.classifier.train_model(x=train_cutouts, y=train_labels, **model_fit_kwargs)

    def default_init_and_train_model(self) -> None:

        self._check_categorized()
        train_test_split = 0.8
        all_labeled_cutouts = self._get_all_labeled_cutouts(shuffle_cutouts=True)
        split_index = int(train_test_split * len(all_labeled_cutouts["labels"]))
        train_labels = all_labeled_cutouts["labels"][:split_index]
        train_cutouts = all_labeled_cutouts["cutouts"][:split_index]

        self.classifier.create_default_tf_keras_model(len(train_cutouts[0]))
        self.classifier.default_compile_model()
        self.classifier.default_train_model(train_cutouts, train_labels)

    def evaluate_model(
        self,
        test_spikes: np.ndarray,
        test_labels: np.ndarray,
        **confusion_matrix_kwargs,
    ) -> Dict[str, Any]:
        """Get statistical measurements for this model

        Parameters
        ----------
        test_spikes : np.ndarray
            spikes used to test and obtain measurements
        test_labels : np.ndarray
            corresponding spike labels used to teest and obtain measurements
        **confusion_matrix_kwargs
            keyworded arguments for sklearn.metrics.confusion_matrix()

        Returns
        -------
        accuracy : float
        precision : float
        recall : float
        f1 : float
        """

        self._check_classifier_initiated()
        self._check_categorized()
        conf_matrix = self.classifier.get_confusion_matrix(
            test_spikes, test_labels, **confusion_matrix_kwargs
        )
        tp = conf_matrix[0][0]
        fn = conf_matrix[0][1]
        fp = conf_matrix[1][0]
        tn = conf_matrix[1][1]

        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * (precision * recall) / (precision + recall)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # Below is work in progress

    # def get_only_neuronal_spikes(
    #     self,
    #     exp_data: Data,
    #     accuracy_threshold: float = 0.9,
    #     signal_filter: Optional[FilterProtocol] = None,
    #     spike_detector: Optional[SpikeDetectionProtocol] = None,
    # ) -> List[SpikestampsType]:
    #     """This method takes each signle individual spike in the experiment data
    #     and uses the trained model to predict whether the spike is neuronal.

    #     Parameters
    #     ----------
    #     exp_data : Data
    #         Experiment data
    #     accuracy_threshold : float, defualt = 0.9
    #         The category prediction comes in probabilities for each category.
    #         If the probability for "neuronal" is higher than this threshold,
    #         the spike will be included.
    #     signal_filter : Optional[FilterProtocol], default = None
    #         Filter applied to the experiment data before spike detection.
    #         If left empty or None, the same filter for the spontaneous data
    #         will be used.
    #     spike_detector : Optional[SpikeDetectionProtocol], default = None
    #         The spike detector used to detect spikes on the experiment data.
    #         If left empty or None, the same spike detector for the spontaneous
    #         data will be used.

    #     Returns
    #     -------
    #     (Same return format as SpikeDetectionProtocol)
    #     A list of SpikestampsType for each channel
    #     """
    #     if not self.trained:
    #         raise Exception("Abnormality detector is not trained yet.")

    #     exp_filter = signal_filter if signal_filter else self.spont_signal_filter
    #     exp_detector = spike_detector if spike_detector else self.spont_spike_detector
    #     list_of_cutout_channels = self._get_cutouts(
    #         exp_data, exp_filter, exp_detector, self.extractor
    #     )
    #     new_spiketrains: List[SpikestampsType] = []

    #     prob_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
    #     for chan_index, cutout_channel in tqdm(enumerate(list_of_cutout_channels)):
    #         times = []
    #         t_stop = 0
    #         for cutout_index, spike_cutout in enumerate(cutout_channel.cutouts):
    #             prediction = prob_model.predict(
    #                 np.array([spike_cutout.cutout]), verbose=0
    #             )
    #             if prediction[0][0] >= accuracy_threshold:
    #                 times.append(spike_cutout.time)
    #                 t_stop = spike_cutout.time
    #         new_spiketrains.append(SpikeTrain(times, t_stop, pq.s))

    #     return new_spiketrains
