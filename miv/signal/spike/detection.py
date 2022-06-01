__doc__ = """

Spike Detection
###############

Code Example::

    detection = ThresholdCutoff(cutoff=3.5)
    spiketrains = detection(signal, timestamps, sampling_rate)


.. currentmodule:: miv.signal.spike

.. autosummary::
   :nosignatures:
   :toctree: _toctree/DetectionAPI

   SpikeDetectionProtocol
   ThresholdCutoff

"""
__all__ = ["ThresholdCutoff"]

from typing import Iterable, List, Union

from dataclasses import dataclass

import neo
import numpy as np
import quantities as pq
from tqdm import tqdm

from miv.typing import SignalType, SpikestampsType, TimestampsType


@dataclass
class ThresholdCutoff:
    """ThresholdCutoff
    Spike sorting step by step guide is well documented `here <http://www.scholarpedia.org/article/Spike_sorting>`_.

        Attributes
        ----------
        dead_time : float
            (default=0.003)
        search_range : float
            (default=0.002)
        cutoff : Union[float, np.ndarray]
            (default=5.0)
        use_mad : bool
            (default=False)
        tag : str
    """

    dead_time: float = 0.003
    search_range: float = 0.002
    cutoff: float = 5.0
    use_mad: bool = True
    tag: str = "Threshold Cutoff Spike Detection"

    def __call__(
        self,
        signal: SignalType,
        timestamps: TimestampsType,
        sampling_rate: float,
        units: Union[str, pq.UnitTime] = "sec",
        progress_bar: bool = True,
    ) -> List[SpikestampsType]:
        """Execute threshold-cutoff method and return spike stamps

        Parameters
        ----------
        signal : SignalType
            signal
        timestamps : TimestampsType
            timestamps
        sampling_rate : float
            sampling_rate
        units : Union[str, pq.UnitTime]
            (default='sec')
        progress_bar : bool
            Toggle progress bar (default=True)

        Returns
        -------
        spiketrain_list : List[SpikestampsType]

        """
        # Spike detection for each channel
        spiketrain_list = []
        num_channels = signal.shape[1]  # type: ignore
        for channel in tqdm(range(num_channels), disable=not progress_bar):
            array = signal[:, channel]  # type: ignore

            # Spike Detection: get spikestamp
            spike_threshold = self.compute_spike_threshold(
                array, cutoff=self.cutoff, use_mad=self.use_mad
            )
            crossings = self.detect_threshold_crossings(
                array, sampling_rate, spike_threshold, self.dead_time
            )
            spikes = self.align_to_minimum(
                array, sampling_rate, crossings, self.search_range
            )
            spikestamp = spikes / sampling_rate
            # Convert spikestamp to neo.SpikeTrain (for plotting)
            spiketrain = neo.SpikeTrain(
                spikestamp, units=units, t_stop=timestamps.max()
            )
            spiketrain_list.append(spiketrain)
        return spiketrain_list

    def compute_spike_threshold(
        self, signal: SignalType, cutoff: float = 5.0, use_mad: bool = True
    ) -> float:  # TODO: make this function compatible to array of cutoffs (for each channel)
        """
        Returns the threshold for the spike detection given an array of signal.

        Spike sorting step by step, `step 2 <http://www.scholarpedia.org/article/Spike_sorting>`_.

        Parameters
        ----------
        signal : np.array
            The signal as a 1-dimensional numpy array
        cutoff : float
            The spike-cutoff multiplier. (default=5.0)
        use_mad : bool
            Noise estimation method. If set to false, use standard deviation for estimation. (default=True)
        """
        if use_mad:
            noise_mid = np.median(np.absolute(signal)) / 0.6745
        else:
            noise_mid = np.std(signal)
        spike_threshold = -cutoff * noise_mid
        return spike_threshold

    def detect_threshold_crossings(self, signal, fs, threshold, dead_time):
        """
        Detect threshold crossings in a signal with dead time and return them as an array

        The signal transitions from a sample above the threshold to a sample below the threshold for a detection and
        the last detection has to be more than dead_time apart from the current one.

        :param signal: The signal as a 1-dimensional numpy array
        :param fs: The sampling frequency in Hz
        :param threshold: The threshold for the signal
        :param dead_time: The dead time in seconds.
        """
        dead_time_idx = dead_time * fs
        threshold_crossings = np.diff((signal <= threshold).astype(int) > 0).nonzero()[
            0
        ]
        distance_sufficient = np.insert(
            np.diff(threshold_crossings) >= dead_time_idx, 0, True
        )
        while not np.all(distance_sufficient):
            # repeatedly remove all threshold crossings that violate the dead_time
            threshold_crossings = threshold_crossings[distance_sufficient]
            distance_sufficient = np.insert(
                np.diff(threshold_crossings) >= dead_time_idx, 0, True
            )
        return threshold_crossings

    def get_next_minimum(self, signal, index, max_samples_to_search):
        """
        Returns the index of the next minimum in the signal after an index

        :param signal: The signal as a 1-dimensional numpy array
        :param index: The scalar index
        :param max_samples_to_search: The number of samples to search for a minimum after the index
        """
        search_end_idx = min(index + max_samples_to_search, signal.shape[0])
        min_idx = np.argmin(signal[index:search_end_idx])
        return index + min_idx

    def align_to_minimum(self, signal, fs, threshold_crossings, search_range):
        """
        Returns the index of the next negative spike peak for all threshold crossings

        :param signal: The signal as a 1-dimensional numpy array
        :param fs: The sampling frequency in Hz
        :param threshold_crossings: The array of indices where the signal crossed the detection threshold
        :param search_range: The maximum duration in seconds to search for the minimum after each crossing
        """
        search_end = int(search_range * fs)
        aligned_spikes = [
            self.get_next_minimum(signal, t, search_end) for t in threshold_crossings
        ]
        return np.array(aligned_spikes)
