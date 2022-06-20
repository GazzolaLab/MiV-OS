## Welch PSD for an electrode's Signal
## Welch Coherence Estimation between signal X and Y


import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import coherence, csd, welch

from miv.typing import SignalType


def plot_spectral(
    signal: SignalType, X: float, Y: float, sampling_rate: float, Number_Segments: float
):

    # Plots Power Spectral Densities for channels X and Y, Cross Power Spctral Densities and Coherence between them
    # Parameters
    # ----------
    # signal : SignalType
    #    Input signal
    # X : float
    #    First Channel
    # Y : float
    #    Second Channel
    # sampling_rate : float
    #    Sampling frequency
    # Number_Segments: float
    # Number of segments to divide the entire signal

    # Returns
    # -------
    # figure: plt.Figure
    # axes

    L = np.int32(len(signal[:, 0]) / Number_Segments)  # L = length of each segment
    fs = sampling_rate  # fs = Sampling Frequeny

    fx, Pxx_den = welch(signal[:, X], fs, nperseg=L)  # Welch PSD and frequency for X
    fy, Pyy_den = welch(signal[:, Y], fs, nperseg=L)  # Welch PSD and frequency  for Y
    fxy, Pxy = csd(
        signal[:, X], signal[:, Y], fs, nperseg=L
    )  # Welch CSD and frequency for X and Y
    fcxy, Cxy = coherence(
        signal[:, X], signal[:, Y], fs, nperseg=L
    )  #  Welch Coherence and frequency for X and Y

    # Plotting

    fig, axes = plt.subplots(2, 2)
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6
    )
    axes[0, 0].semilogy(fx, Pxx_den)
    axes[0, 1].semilogy(fy, Pyy_den)
    axes[1, 0].semilogy(fxy, Pxy)
    axes[1, 1].semilogy(fcxy, Cxy)

    # axes[:,:].set_xlabel('Electrode')
    # axes.set_ylabel('Electrode')
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
