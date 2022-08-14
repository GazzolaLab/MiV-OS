__all__ = ["BensSpikerAlgorithm"]
from typing import Tuple

import numpy as np
from scipy.signal import firwin
from sklearn.preprocessing import normalize

from miv.typing import SignalType, TimestampsType


class BensSpikerAlgorithm:
    """
    Bens Spiker Algorithm (BSA) Implementation.
    Implementation based on [1]_ [2]_ [3]_

    .. [1] Petro et al. (2020)
    .. [2] Sengupta et al. (2017)
    .. [3] Schrauwen, B., et al., BSA, a fast and accurate spike train encoding scheme. IJCNN (2003)

    Examples
    --------

        >>> bsa_spiker: SpikerAlgorithmProtocol  = BensSpikerAlgorithm()
        >>> bsa_spiker(data)
        [[[0 1 1 ... 1 0 0]
          [0 1 1 ... 0 0 1]
          ...
          [0 0 0 ... 0 1 0]]
         [[0 0 1 ... 1 0 1]
          [1 0 1 ... 0 1 1]
          ...
          [0 0 1 ... 1 0 0]]]
    """

    def __init__(
        self,
        sampling_rate: float,
        threshold: float = 1.0,
        fir_filter_length: int = 2,
        fir_cutoff: float = 0.8,
        normalize: bool = False,
    ):
        """

        Parameters
        ----------
        sampling_rate : float
            Sampling rate in float. Used to compute timestamps to return.
            Note, this might not be the sampling rate of the signal, if signal passed
            through auditory model.
        threshold : float or Iterable[float]
            Threshold level for spike. It can also be a list of threshold.
            If list of threshold is given, the length of the list must be
            the number of features/channels. (default=1.0)
        fir_filter_length : int
            Filter_length for finite impulse response (FIR) filter (default=2)
        fir_cutoff : float
            Cutoff for finite impulse response (FIR) filter (default=0.8)
        normalize : bool
            Normalize data before comparing against threshold. (default=False)
            If true, input data is normalized in range of [0,1].
            IF false, input data is only shifted to have range of [0,~]
        """
        super().__init__()
        self.threshold = threshold
        self.fir_filter_length = fir_filter_length
        self.fir_cutoff = fir_cutoff
        self.data_normalize = normalize
        self.sampling_rate = sampling_rate

    def __call__(self, data: np.ndarray, time_offset: float = 0) -> np.ndarray:
        """
        Returns spiketrain after applying BSA to the data.

        Parameters
        ----------
        data : np.ndarray
            shape (length, n_channels)
        time_offset : float
            time offset for returning timestamps (default=0)

        Returns
        -------
        Spiketrains : SpiketrainType
        timestamps: TimestampsType

        """

        # Data prep
        assert (
            len(data.shape) == 2
        ), "The shape of the data must be (length, n_channels)."
        data_length, num_channels = data.shape

        # Set thresholds
        if isinstance(self.threshold, list):
            assert num_channels == len(
                self.threshold
            ), "If threshold is given as a list, \
                    the length of the threshold must be same as the number of channels in data."
        elif isinstance(self.threshold, float) or isinstance(self.threshold, int):
            thresholds = np.ones(num_channels) * self.threshold
        else:
            raise TypeError(
                f"The threshold type {type(self.threshold)} is not supported."
            )

        # FIR
        filter_values, filter_length = self._finite_impulse_response(data)

        # Data standardization
        if self.data_normalize:
            _data = normalize(data, axis=1)
        else:
            _data = data.copy()
            _data -= data.min(axis=0, keepdims=True)

        spikes = np.zeros_like(data, dtype=np.bool_)

        # BSA Algorithm
        for ch in range(0, num_channels):  # TODO: Multithreaded
            for i in range(0, data_length - filter_length + 1):
                error1 = 0
                error2 = 0
                for j in range(0, filter_length):
                    error1 += abs(_data[i + j][ch] - filter_values[j])
                    error2 += abs(_data[i + j - 1][ch])

                if error1 <= error2 * thresholds[ch]:
                    spikes[i][ch] = True
                    for j in range(0, filter_length):
                        if i + j + 1 < data_length:
                            _data[i + j + 1][ch] -= filter_values[j]

        # Timestamps
        timestamps = np.arange(data_length) * (1 / self.sampling_rate) + time_offset

        return spikes, timestamps

    def _finite_impulse_response(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        FIR filter

        Returns
        -------
        filtered_value
        filtered_length
        """
        filter_values = firwin(self.fir_filter_length, cutoff=self.fir_cutoff)
        return filter_values, len(filter_values)
