__all__ = ["plot_frequency_domain"]

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from scipy.signal import welch

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
    plt.plot(np.abs(sig_fft) ** 2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("DFT frequency")

    # Welch (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html)
    f, Pxx_den = welch(signal, sampling_rate, nperseg=1024)
    f_med, Pxx_den_med = welch(signal, sampling_rate, nperseg=1024, average="median")
    plt.figure()
    plt.semilogy(f, Pxx_den, label="mean")
    plt.semilogy(f_med, Pxx_den_med, label="median")
    plt.xlabel("frequency [Hz]")
    plt.ylabel("PSD [uV**2/Hz]")
    plt.legend()
    return fig
