#Import required modules

import os, sys
import miv
from miv.io import DataManager
from miv.signal.filter import ButterBandpass, MedianFilter, FilterCollection
from miv.signal.spike import ThresholdCutoff, SpikeSorting, PCADecomposition
from miv.statistics import signal_to_noise, firing_rates
from miv.visualization import plot_frequency_domain, extract_waveforms, plot_waveforms
from miv.io import file as miv_file

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

recording_id = "2022-03-10_16-19-09-Record Node 104-experiment1_spontaneous-recording1"
input_data, data_container = miv_file.load('2022-03-10_16-19-09/MiV_data.h5', 
                                          datasets=recording_id)

signal = input_data[f"{recording_id}/signal"]
timestamps = input_data[f"{recording_id}/timestamps"]
sampling_rate = input_data[f"{recording_id}/sampling_rate"][0]
num_channel = signal.shape[-1]

#Set up filters, we use butter bandpass and median filters here. More details are here (https://miv-os.readthedocs.io/en/latest/api/signal.html)
signal_filter = (
    FilterCollection()
        .append(ButterBandpass(300, 3000, order=4))     
        .append(MedianFilter(threshold=60, k=20))
)

#Set threshold for Signal to noise ratio to detect spikes
spike_detection = ThresholdCutoff(cutoff=5)

# Preprocess
signal = signal_filter(signal[0], sampling_rate)
spiketrains = spike_detection(signal, timestamps, sampling_rate)

#Estimate Firing Rate
stat = firing_rates(spiketrains)['rates']

#Plot rasterplot using viziphant 
from viziphant.rasterplot import rasterplot_rates
plt.figure(figsize=(24,8))
a,b,c = rasterplot_rates(spiketrains, ax=plt.gca())

