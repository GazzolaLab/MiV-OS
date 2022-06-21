__all__ = ["plot_frequency_domain", "plot_spectral"]

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from scipy.signal import coherence, csd, welch

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


def plot_spectral(
    signal: SignalType, X: float, Y: float, sampling_rate: float, Number_Segments: float
):
    """
    Plots Power Spectral Densities for channels X and Y, Cross Power Spctral Densities and Coherence between them

    Parameters
    ----------
    signal : SignalType
        Input signal
    X : float
        First Channel
    Y : float
        Second Channel
    sampling_rate : float
        Sampling frequency
    Number_Segments: float
        Number of segments to divide the entire signal

    Returns
    -------
    figure: matplotlib.pyplot.Figure
    axes: matplotlib.Axes
    """
    ## Welch PSD for an electrode's Signal
    ## Welch Coherence Estimation between signal X and Y

    L = np.int32(len(signal[:, 0]) / Number_Segments)  # L = length of each segment
    fs = sampling_rate  # fs = Sampling Frequeny

    fx, Pxx_den = welch(signal[:, X], fs, nperseg=L)  # Welch PSD and frequency for X
    fy, Pyy_den = welch(signal[:, Y], fs, nperseg=L)  # Welch PSD and frequency  for Y
    fxy, Pxy = csd(  # Welch CSD and frequency for X and Y
        signal[:, X], signal[:, Y], fs, nperseg=L
    )
    fcxy, Cxy = coherence(  # Welch Coherence and frequency for X and Y
        signal[:, X], signal[:, Y], fs, nperseg=L
    )

    # Plotting
    fig, axes = plt.subplots(2, 2)
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6
    )
    axes[0, 0].semilogy(fx, Pxx_den)
    axes[0, 1].semilogy(fy, Pyy_den)
    axes[1, 0].semilogy(fxy, Pxy)
    axes[1, 1].semilogy(fcxy, Cxy)

    axes[0, 0].set_ylabel("PSD [V**2/Hz]")
    axes[1, 0].set_ylabel("CSD [V**2/Hz]")
    axes[1, 0].set_xlabel("'frequency [Hz]'")
    axes[1, 1].set_xlabel("'frequency [Hz]'")
    axes[0, 1].set_ylabel("PSD [V**2/Hz]")
    axes[1, 1].set_ylabel("Coherence")
    axes[0, 0].set_xlabel("'frequency [Hz]'")
    axes[0, 1].set_xlabel("'frequency [Hz]'")
    axes[0, 0].set_title("PSD for X")
    axes[0, 1].set_title("PSD for Y")
    axes[1, 0].set_title("CPSD for X and Y")
    axes[1, 1].set_title("Coherence for X,Y")
    return fig, axes
