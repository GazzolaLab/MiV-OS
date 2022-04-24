import os
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.signal import lfilter, savgol_filter
from scipy import fftpack

# MCS PyData tools
# import McsPy
# import McsPy.McsData
# from McsPy import ureg, Q_

# VISUALIZATION TOOLS
import matplotlib.pyplot as plt


def plot_frequency_domain(signal, sampling_rate=1):
    # FFT
    fig = plt.figure()
    sig_fft = fftpack.fft(signal)
    power = np.abs(sig_fft) ** 2
    # sample_freq = fftpack.fftfreq(signal.size, d=1 / sampling_rate)
    plt.plot(power)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("power")
    return fig
