---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Burst Analysis

Bursting is defined as the occurence of a specified number of simultaneuos spikes (usually >10), with a small interspike interval (usually < 100ms) [1,2]

References: 

[1] Chiappalone, Michela, et al. "Burst detection algorithms for the analysis of spatio-temporal patterns in cortical networks of neurons." Neurocomputing 65 (2005): 653-662.\

[2] Eisenman, Lawrence N., et al. "Quantification of bursting and synchrony in cultured hippocampal neurons." Journal of neurophysiology 114.2 (2015): 1059-1071.

+++

## 1. Data Load and Preprocessing

```{code-cell} ipython3
:tags: [hide-cell]

import os, sys
import miv
import scipy.signal as ss

from miv.io import DataManager
from miv.signal.filter import ButterBandpass, MedianFilter, FilterCollection
from miv.signal.spike import ThresholdCutoff
from miv.statistics import pairwise_causality
from miv.visualization import plot_spectral, pairwise_causality_plot
import numpy as np
from miv.typing import SignalType
```

```{code-cell} ipython3
# Experiment name
experiment_query = "experiment0"

# Data call
signal_filter = (
    FilterCollection()
        .append(ButterBandpass(600, 2400, order=4))
        .append(MedianFilter(threshold=60, k=30))
)
spike_detection = ThresholdCutoff(cutoff=5)

# Spike Detection
data_collection = optogenetic.load_data()
data = data_collection.query_path_name(experiment_query)[0]
#false_channels = [12,15,36,41,42,43,45,47,53,57,55,58,61,62]
#data.set_channel_mask(false_channels)
with data.load() as (signal, timestamps, sampling_rate):
    # Preprocess
    signal = signal_filter(signal, sampling_rate)
    spiketrains = spike_detection(signal, timestamps, sampling_rate)
```

## 2. Burst Estimations
     
     Function Call
    
     burst(spiketrains: SpikestampsType, channel: float, min_isi: float, min_len: float)

     Calculates parameters critical to characterize bursting phenomenon on a single channel

     Parameters
     ----------
     spikes : SpikestampsType
            Single spike-stamps
     Channel : float
        Channel to analyze
     min_isi : float
        Minimum Interspike Interval (in seconds) to be considered as bursting [standard = 0.1]
     min_len : float
        Minimum number of simultaneous spikes to be considered as bursting [standard = 10]

     Returns
     -------
     start_time: float
     The time instances when a burst starts

     burst_duration: float
     The time duration of a particular burst

     burst_len: float
     Number of spikes in a particular burst

     burst_rate: float
     firing rates corresponding to particular bursts
    


```{code-cell} ipython3
#Example
burst(spiketrains,45,0.1,10)
# Estimates the burst parameters for 45th electrode with bursts defined as more than 10 simultaneous spikes with 0.1 s interspike interval 
```

## 3. Plot bursting events across electrodes

     Function Call:
     
     plot_burst(spiketrains: SpikestampsType, min_isi: float, min_len: float)
     
     Parameters
     ----------
     spikes : SpikestampsType
            Single spike-stamps
     min_isi : float
        Minimum Interspike Interval (in seconds) to be considered as bursting [standard = 0.1]
     min_len : float
        Minimum number of simultaneous spikes to be considered as bursting [standard = 10]

     Returns
     -------
     figure, axes
     matplot figure with bursts plotted for all electrodes


```{code-cell} ipython3
#Example
plot_burst(spiketrains,0.1,10)
# plots the burst events fo with bursts defined as more than 10 simultaneous spikes with 0.1 s interspike interval 
```

## Welch Coherence 

Plots Power Spectral Densities for channels X and Y, Cross Power Spctral Densities and Coherence between them using Welch's method\

plot_spectral(signal, X, Y, sampling_rate, Number_Segments)\


Parameters\
#----------
signal : SignalType\
    Input signal\
X : float\
    First Channel \
Y : float\
    Second Channel\
sampling_rate : float\
    Sampling frequency\
Number_Segments: float\
Number of segments to divide the entire signal\

Returns\
#-------
figure: plt.Figure\
axes\

References:\ 

1) https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html
2) P. Welch, “The use of the fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms”, IEEE Trans. Audio Electroacoust. vol. 15, pp. 70-73, 1967.


```{code-cell} ipython3
#Example 
plot_spectral(signal,1,42,30000,10000)

##Plots the PSDs, CPSD and Coherence for channel 1 & 43 for a sampling rate of 30000, with signal divided into 10000 segments
```
