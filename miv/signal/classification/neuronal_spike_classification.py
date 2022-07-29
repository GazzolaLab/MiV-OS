__all__ = ["NeuronalSpikeClassifier"]

from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

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

    def create_default_tf_keras_model(self, input_size: int) -> None:
        """Creates a defualt classification model

        Parameters
        ----------
        input_size : int
            The input size that determines the number of nodes in the input
            layer and the hidden layer.
        """

        layers = [
            tf.keras.layers.Dense(input_size),
            tf.keras.layers.Dense(
                # This needs to be tweaked
                input_size
            ),
            tf.keras.layers.Dense(1, activation="sigmoid"),
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

        self.model.compile(
            optimizer="Adam",
            loss="BinaryCrossentropy",
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

    def default_train_model(self, spikes: np.ndarray, labels: np.ndarray) -> None:
        cb = tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=2)
        self.model.fit(x=spikes, y=labels, callbacks=cb, epochs=5)

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

        predictions = self.predict_categories_sigmoid(test_spikes, verbose=0)
        predictions = (predictions + 1) % 2
        test_labels = (test_labels + 1) % 2
        return confusion_matrix(test_labels, predictions, **confusion_matrix_kwargs)

    def predict_categories_sigmoid(
        self, spikes: np.ndarray, threshold: float = 0.5, **predict_kwargs
    ) -> np.ndarray:
        """Predict with spikes and get the index of category for each prediction

        Parameters
        ----------
        spikes : np.ndarray
        threshold : float, default = 0.5
            Prediction values above this threshold value will be marked as 1.
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
            outcomes[i] = prediction > threshold
        return outcomes

    def _validate_model(self):
        try:
            self.model
        except AttributeError:
            raise Exception("model is not set yet")
