import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

from miv.visualization.fft_domain import plot_frequency_domain, plot_spectral


def test_plot_frequency_domain():
    signal_length = 32
    signal = np.random.rand(signal_length)
    ref_sig_fft = fftpack.fft(signal)
    ref_sig_fft_magnitude = np.abs(ref_sig_fft) ** 2
    fig = plot_frequency_domain(signal)
    ax = fig.gca()

    assert ax.get_xlabel() == "Frequency [Hz]"
    assert ax.get_ylabel() == "DFT frequency"
    test_sig_fft_magntiude = ax.lines[0].get_ydata()
    np.testing.assert_allclose(ref_sig_fft_magnitude, test_sig_fft_magntiude)


def test_plot_spectral_axes_annotation():
    signal_length = 32
    num_channel = 2
    np_rng = np.random.default_rng(0)
    signal = np_rng.random([signal_length, num_channel])
    fig, axes = plot_spectral(signal, 0, 1, 1, 6)

    assert isinstance(fig, plt.Figure)
    assert axes[0, 0].get_xlabel() == "Frequency [Hz]"
    assert axes[0, 1].get_xlabel() == "Frequency [Hz]"
    assert axes[1, 0].get_xlabel() == "Frequency [Hz]"
    assert axes[1, 1].get_xlabel() == "Frequency [Hz]"
    assert axes[0, 0].get_ylabel() == "PSD [V**2/Hz]"
    assert axes[0, 1].get_ylabel() == "PSD [V**2/Hz]"
    assert axes[1, 0].get_ylabel() == "CSD [V**2/Hz]"
    assert axes[1, 1].get_ylabel() == "Coherence"
    assert axes[0, 0].get_title() == "PSD for X"
    assert axes[0, 1].get_title() == "PSD for Y"
    assert axes[1, 0].get_title() == "CPSD for X and Y"
    assert axes[1, 1].get_title() == "Coherence for X,Y"
