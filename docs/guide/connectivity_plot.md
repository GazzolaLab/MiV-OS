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

# Channel-wise Signal Correlation

Here is the example script of cross-correlation analysis using `Elephant` package.

```{code-cell} ipython3
:tags: [hide-cell]

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss

import miv
from miv.datasets import optogenetic
from miv.io import DataManager
from miv.signal.filter import ButterBandpass, FilterCollection, MedianFilter
from miv.signal.spike import ThresholdCutoff
from miv.visualization.connectivity_plot import plot_connectivity, plot_connectivity_plot

```

## Pre-processing

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

## Provide MEA mao

```{code-cell} ipython3
#Matrix containing electrode numbers according to their spatial location
mea_map = np.array([[25  , 10 , 12 , 14 , 31 , 28 , 26 , 40],
                         [18 , 17 , 11 , 13 , 32 , 27 , 38 , 37],
                         [20 , 19 , 9  , 15 , 30 , 39 , 36 , 35],
                         [23 , 22 , 21 , 16 , 29 , 34 , 33 , 56],
                         [24 , 1  , 2  , 61 , 44 , 53 , 54 , 55],
                         [3  , 4  , 7  , 62 , 43 , 48 , 51 , 52],
                         [5  , 6  , 59 , 64 , 41 , 46 , 49 , 50],
                         [57 , 58 , 60 , 63 , 42 , 45 , 47 , 8]])

```

## Forming Connectivity Matrix

```{code-cell} ipython3
#Correlation using Elephant
import elephant.statistics
import quantities as pq
from elephant.spike_train_correlation import correlation_coefficient

rates = elephant.statistics.BinnedSpikeTrain(spiketrains, bin_size=2*pq.ms)
corrcoef_matrix = correlation_coefficient(rates)

```

## Plot Connectivity

Plots connectivity using the provided MEA map and connectivity matrix. Documentation is available [here](miv.visualization.connectivity_plot.plot_connectivity)

```{code-cell} ipython3
#Non-interactive connectivity plot using correlation
plot_connectivity(mea_map, corrcoef_matrix, False)

```

```{code-cell} ipython3
#Interactive connectivity plot using correlation
plot_connectivity_interactive(mea_map, corrcoef_matrix, False)

```
