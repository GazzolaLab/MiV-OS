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

Bursting is defined as the occurence of a specified number of simultaneuos spikes (usually >10), with a small interspike interval (usually < 100ms) [1][1],[2][2]

[1]: https://www.sciencedirect.com/science/article/abs/pii/S0925231204004874
[2]: https://journals.physiology.org/doi/full/10.1152/jn.00079.2015?rfr_dat=cr_pub++0pubmed&url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org

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
Calculates parameters critical to characterize bursting phenomenon on a single channel. Documentation is available [here](miv.statistics.burst).

```{code-cell} ipython3
# Estimates the burst parameters for 45th electrode with bursts defined as more than 10 simultaneous spikes with 0.1 s interspike interval 
burst(spiketrains,45,0.1,10)
```

## 3. Plotting
Plots the burst events across the recordings. Documentation is available [here](miv.visualization.plot_burst).

```{code-cell} ipython3
#Example
# plots the burst events with bursts defined as more than 10 simultaneous spikes with 0.1 s interspike interval
plot_burst(spiketrains,0.1,10)
 
```

