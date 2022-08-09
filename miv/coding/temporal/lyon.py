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
    Lyon's Ear Model. Wrapper for lyon.LyonCalc module.

    Implementation details can be found `here <https://engineering.purdue.edu/~malcolm/interval/1998-010/>`_
    The above MATLAB/C implementation is ported to python `here <https://github.com/sciforce/lyon>`_

    .. automodule:: lyon.LyonCalc.lyon_passive_ear
    """

    def __init__(
        self,
        sampling_rate: float = 16000,
        decimation_factor: int = 1,
        ear_quality: int = 8,
        step_factor: Optional[float] = None,
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
            decimation_factor How much to decimate model output. (default=1)
        ear_quality : int
            Smaller values mean broader filters. (default=8)
        step_factor : float
            Filter stepping factor. If not given, default value set to 25% overlap.
            (default=ear_quality / 32)
        differ : bool
            Differ Channel difference: improves model's freq response. (default=True)
        agc : bool
            Whether to use AGC for neural model adaptation. (default=True)
        tau_factor : int
            Reduces antialiasing in filter decimation. (default=3)
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
        """

        Parameters
        ----------
        signal : np.ndarray
            signal waveform: mono,

        Returns
        -------
        Output : np.ndarray
            array shape (N / decimation_factor, channels)
        """
        return self.lyon.lyon_passive_ear(signal, **self.parameters)
