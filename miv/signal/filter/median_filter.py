__all__ = ["MedianFilter"]

from typing import Optional, Tuple, Union

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from miv.core.datatype import Signal
from miv.core.operator import OperatorMixin
from miv.core.wrapper import wrap_generator_to_generator
from miv.typing import SignalType


@dataclass
class MedianFilter(OperatorMixin):
    """Median filter with threshold

    If the signal exceed the threshold, the value is replaced by median of neighboring
    values.

    The filter is designed to mitigate unusually high amplitude of spike within
    the signal.

    Parameters
    ----------
    threshold : float
        clipping threshold.
    k : Union[int, Tuple[int, int]]
        Number of neighboring values to take median. (default=20)
        It is possible to provide (k1, k2) tuple to separate right-neighbor bound
        and left-neighbor bound. If you are unsure which value to use, try with the
        value at least `k=0.002 * sampling_rate`.
    tag : str
        Tag for the collection of filter.
    """

    threshold: float
    k: Union[int, Tuple[int, int]] = 20
    tag: str = "median filter"

    @wrap_generator_to_generator
    def __call__(
        self,
        signal: SignalType,
    ) -> SignalType:
        """__call__.

        Parameters
        ----------
        signal : SignalType
            pre-filtered signal

        Returns
        -------
        SignalType
            filtered signal

        """
        y = signal.copy()
        if isinstance(self.k, int):
            k = (self.k, self.k)
        else:
            k = self.k
        outlier_i, outlier_ch = np.where(np.abs(signal) > self.threshold)
        for i, ch in zip(outlier_i, outlier_ch):
            low_bound = np.max((0, i - k[0]))
            up_bound = np.min((i + k[1], signal.shape[0])) + 1
            y[low_bound:up_bound, ch] = np.median(signal[low_bound:up_bound, ch])
        return y

    def __post_init__(self):
        assert (
            self.threshold is None or self.threshold > 0.0
        ), f"Filter threshold {self.threshold} must be positive real number."
        if self.threshold is not None and self.threshold < 50.0:
            logging.warning(
                "Threshold is less than 50.0 uV, which could alter the spike signal"
            )
        super().__init__()
        self.cacher.policy = "OFF"
