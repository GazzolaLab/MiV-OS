__all__ = ["NeuronalSpikeClassifier"]

from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from miv.io import Data, DataManager
from miv.signal.events.protocol import SpikeClassificationModelProtocol


class NeuronalSpikeClassifier:
    """Neuronal Spike Detector

    This object uses a deep learning model to tell neuronal spikes apart from other spikes.

        Attributes
        ----------
        model : SpikeClassificationModelProtocol

    """

    def __init__(
        self, model: Optional[SpikeClassificationModelProtocol] = None
    ) -> None:
        self.model: Optional[SpikeClassificationModelProtocol] = model
        self.train_spikes: np.ndarray
        self.train_labels: np.ndarray

    def create_default_tf_keras_model(
        self, train_spikes: np.ndarray, train_labels: np.ndarray
    ) -> None:

        if np.shape(train_spikes)[0] != np.shape(train_labels)[0]:
            raise Exception("train spikes and train labels have incompatible sizes")
        if np.size(train_spikes) == 0 or np.size(train_labels == 0):
            raise Exception("can't create default model from empty training data")

        layers = [
            tf.keras.layers.Dense(np.shape(train_spikes)[1]),
            tf.keras.layers.Dense(
                np.shape(train_spikes)[1] / 2
            ),  # This needs to be tweaked
            tf.keras.layers.Dense(len(np.unique(train_labels))),
        ]

        self.model = tf.keras.Sequential(layers)

    def compile_model(self, **compile_kwargs) -> None:
        self.model.compile(compile_kwargs)

    def default_compile_model(self) -> None:
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train_model(self, **fit_kwargs) -> None:
        self.model.fit(fit_kwargs)

    def get_confusion_matrix(
        self,
        test_spikes: np.ndarray,
        test_labels: np.ndarray,
        **confusion_matrix_kwargs,
    ) -> np.ndarray:

        self._validate_model()

        predictions = self.predict_categories(test_spikes, verbose=0)
        return tf.math.confusion_matrix(
            test_labels, predictions, confusion_matrix_kwargs
        )

    def predict_categories(self, spikes: np.ndarray, **predict_kwargs) -> np.ndarray:

        self._validate_model()

        predictions = self.model.predict(spikes, predict_kwargs)
        outcomes = np.ndarray(np.shape(spikes)[0])
        for i, prediction in enumerate(predictions):
            outcomes[i] = np.argmax(prediction)
        return outcomes

    def _validate_model(self):
        if self.model is None:
            raise Exception("model is not set yet")
