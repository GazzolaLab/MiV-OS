__all__ = ["plot_frequency_domain"]

import os
import numpy as np

from scipy import fftpack

import matplotlib.pyplot as plt

from miv.typing import SignalType


def plot_frequency_domain(signal: SignalType, sampling_rate: float) -> plt.Figure:
    """
    Plot DFT frequency domain

    Parameters
    ----------
    signal : SignalType
        Input signal
    sampling_rate : float
        Sampling frequency

    Returns
    -------
    figure: plt.Figure

    """
    # FFT
    fig = plt.figure()
    sig_fft = fftpack.fft(signal)
    # sample_freq = fftpack.fftfreq(signal.size, d=1 / sampling_rate)
    plt.plot(sig_fft)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("DFT frequency")
    return fig
