__all__ = ["NeuronalSpikeClassifier"]

from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from miv.io import Data, DataManager
from miv.signal.classification.protocol import SpikeClassificationModelProtocol


class NeuronalSpikeClassifier:
    """Spike Classifier Classifier

    This object uses a deep learning model to tell neuronal spikes apart from other spikes.

        Attributes
        ----------
        model : SpikeClassificationModelProtocol, default = None
        train_spikes : np.ndarray
        train_labels : np.ndarray

    """

    def __init__(
        self, model: Optional[SpikeClassificationModelProtocol] = None
    ) -> None:
        if model is not None:
            self.model: SpikeClassificationModelProtocol = model
        self.train_spikes: np.ndarray
        self.train_labels: np.ndarray

    def create_default_tf_keras_model(
        self, train_spikes: np.ndarray, train_labels: np.ndarray
    ) -> None:
        """Creates a defualt classification model

        Parameters
        ----------
        train_spikes : np.ndarray
        train_labels : np.ndarray

        """

        if np.shape(train_spikes)[0] != np.shape(train_labels)[0]:
            raise Exception("train spikes and train labels have incompatible sizes")
        if np.size(train_spikes) == 0 or np.size(train_labels) == 0:
            raise Exception("can't create default model from empty training data")

        layers = [
            tf.keras.layers.Dense(np.shape(train_spikes)[1]),
            tf.keras.layers.Dense(
                int(np.shape(train_spikes)[1])
            ),  # This needs to be tweaked
            tf.keras.layers.Dense(len(np.unique(train_labels))),
        ]

        self.model = tf.keras.Sequential(layers)

    def compile_model(self, **compile_kwargs) -> None:
        """Compile model

        Parameters
        ----------
        **compile_kwargs
            keyworded arguments for the compile method of the model
        """
        self.model.compile(**compile_kwargs)

    def default_compile_model(self) -> None:
        """Compile model with default compile paramters

        See docs/discussion/spike_classification/Classifier Model Comparison with Keras Optimizers and Losses.ipynb
        """
        # These need to be tweaked
        self.model.compile(
            optimizer="Adamax",
            loss="SquaredHinge",
            metrics=["accuracy"],
        )

    def train_model(self, **fit_kwargs) -> None:
        """Train model

        Parameters
        ----------
        **fit_kwargs
            keyworded arguments for model.fit method
        """
        self.model.fit(**fit_kwargs)

    def get_confusion_matrix(
        self,
        test_spikes: np.ndarray,
        test_labels: np.ndarray,
        **confusion_matrix_kwargs,
    ) -> np.ndarray:
        """Get a confusion model with test spikes and labels

        Parameters
        ----------
        test_spikes : np.ndarray
        test_labels : np.ndarray
        **confusion_matrix_kwargs
            keyworded arguments for tf.math.confusion_matrix method

        Returns
        -------
        Numpy 2D array of confusion matrix
        """

        self._validate_model()

        predictions = self.predict_categories(test_spikes, verbose=0)
        return tf.math.confusion_matrix(
            test_labels, predictions, **confusion_matrix_kwargs
        )

    def predict_categories(self, spikes: np.ndarray, **predict_kwargs) -> np.ndarray:
        """Predict with spikes and get the index of category for each prediction

        Parameters
        ----------
        spikes : np.ndarray
        **predict_kwargs
            keyworded arguments for model.predict function

        Returns
        -------
        Numpy array of prediction outcomes
        """

        self._validate_model()

        predictions = self.model.predict(spikes, **predict_kwargs)
        outcomes: np.ndarray = np.ndarray(np.shape(spikes)[0])
        for i, prediction in enumerate(predictions):
            outcomes[i] = np.argmax(prediction)
        return outcomes

    def _validate_model(self):
        try:
            self.model
        except AttributeError:
            raise Exception("model is not set yet")
