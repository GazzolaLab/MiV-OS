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
file_format: mystnb
mystnb:
  execution_mode: 'off'
---

# Channel-wise Correlation

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

## Connectivity: Cell Assembly Detection (Elephant)

```{code-cell} ipython3
:tags: [hide-cell]

import elephant.statistics
import quantities as pq
import viziphant
from elephant.cell_assembly_detection import cell_assembly_detection
from elephant.spike_train_correlation import correlation_coefficient
from viziphant.spike_train_correlation import plot_corrcoef

```

```{code-cell} ipython3
#Cell_Assembly_Detection
rates = elephant.statistics.BinnedSpikeTrain(spiketrains, bin_size=10*pq.ms)
rates.rescale('ms')
patterns = cell_assembly_detection(rates, max_lag=5)

ax = viziphant.patterns.plot_patterns(spiketrains, patterns=patterns[:10],
                                 circle_sizes=(1, 10, 10))

ax.set_xlim(0,5)
```

([png](https://uofi.box.com/shared/static/fmppqvg9lydbxlpc7lwr6sniisp3zy4v.png))

![Cell Assembly Detection](https://uofi.box.com/shared/static/fmppqvg9lydbxlpc7lwr6sniisp3zy4v.png)

+++

## Cross Correlation Matrix (Elephant)

```{code-cell} ipython3
#Cross_Correlation Matrix
corrcoef_matrix = correlation_coefficient(rates)

fig, axes = plt.subplots(figsize=(16,8))
plot_corrcoef(corrcoef_matrix, axes=axes)
axes.set_xlabel('Electrode')
axes.set_ylabel('Electrode')
axes.set_title("Correlation coefficient matrix")
```
