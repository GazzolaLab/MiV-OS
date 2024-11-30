__doc__ = "Downsample Operator for Signals"
__all__ = ["Downsample"]

import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.signal as sps

from miv.core.datatype import Signal
from miv.core.operator_generator.operator import GeneratorOperatorMixin
from miv.core.operator_generator.wrapper import cache_generator_call
from miv.typing import SignalType


@dataclass
class Downsample(GeneratorOperatorMixin):
    """Downsample the signal to the target sampling rate, wrapped in operator mixin.

    Parameters
    ----------
    target_rate : float
        Target sampling rate (Hz) for downsampling.
    """

    target_rate: float = 1000.0  # Hz
    tag: str = "downsample operator"

    @cache_generator_call
    def __call__(self, signal: Signal) -> Signal:
        """Downsample the signal to the target sampling rate.

        Parameters
        ----------
        signal : SignalType
            signal
        """
        # Calculate downsampling factor
        downsample_factor = int(signal.rate / self.target_rate)
        if downsample_factor < 1:
            raise ValueError(
                "Invalid downsample factor. Check target and original rates."
            )

        # Downsample each channel
        data = signal.data
        downsampled_data = sps.resample(
            data, num=data.shape[0] // downsample_factor, axis=0
        )

        # Adjust timestamps for the new sampling rate
        downsampled_timestamps = np.linspace(
            signal.timestamps[0], signal.timestamps[-1], downsampled_data.shape[0]
        )

        return Signal(
            data=downsampled_data,
            timestamps=downsampled_timestamps,
            rate=self.target_rate,
        )

    def __post_init__(self):
        super().__init__()
        # self.cacher.policy = "OFF"
