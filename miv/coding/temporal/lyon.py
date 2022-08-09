__doc__ = """
Lyon Ear Model

.. automodule:: LyonEarModel

"""
__all__ = ["LyonEarModel"]

from typing import Optional

import lyon.calc
import numpy as np


class LyonEarModel:  # TemporalEncoderProtocol
    """
    Lyon Ear Model. Wrapper for the module lyon.LyonCalc module.

    Implementation details can be found `here <https://engineering.purdue.edu/~malcolm/interval/1998-010/>`_

    .. automodule:: lyon.LyonCalc.lyon_passive_ear
    """

    def __init__(
        self,
        sampling_rate: float = 16000,
        decimation_factor: int = 1,
        ear_quality: int = 8,
        step_factor: bool = None,
        differ: bool = True,
        agc: bool = True,
        tau_factor: int = 3,
    ):
        """
        Auditory nerve response, based on Lyon's model.

        Parameters
        ----------
        sampling_rate : float
            sample_rate Waveform sample rate. (default=16000)
        decimation_factor : int
            decimation_factor
        ear_quality : int
            ear_quality
        step_factor : bool
            step_factor
        differ : bool
            differ
        agc : bool
            agc
        tau_factor : int
            tau_factor
        """
        """
        @parameter decimation_factor How much to decimate model output. Default: 1
        @parameter ear_q Ear quality. Smaller values mean broader filters. Default: 8
        @parameter step_factor Filter stepping factor. Defaults to ear_q / 32 (25% overlap)
        @parameter differ Channel difference: improves model's freq response. Default: True
        @parameter agc Whether to use AGC for neural model adaptation. Default: True
        @parameter tau_factor Reduces antialiasing in filter decimation. Default: 3
        @returns ndarray of shape [N / decimation_factor, channels]
        """
        self.lyon = lyon.calc.LyonCalc()
        self.parameters = {
            "sample_rate": sampling_rate,
            "decimation_factor": decimation_factor,
            "ear_q": ear_quality,
            "step_factor": step_factor,
            "differ": differ,
            "agc": agc,
            "tau_factor": tau_factor,
        }

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """__call__.

        Parameters
        ----------
        signal : np.ndarray
            signal waveform: mono, float64
        """
        return self.lyon.lyon_passive_ear(signal, **self.parameters)
