import numpy as np
from scipy import fftpack

from miv.visualization.fft_domain import plot_frequency_domain


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
