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

# Signal Processing Guideline

```{code-cell} ipython3
:tags: [hide-cell]

import os
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt

```

## 1. Data Load

```{code-cell} ipython3
:tags: [hide-cell]

from miv.io import load_data
from miv.io.data import Data, Dataset
```

```{code-cell} ipython3
# Load dataset from OpenEphys recording
folder_path: str = "~/Open Ephys/2022-03-10-16-19-09"  # Data Path
# Provide the path of experimental recording tree to the DataSet class
# Data set class will load the data and create a list of objects for each data
# dataset = load_data(folder_path, device="OpenEphys")
dataset = Dataset(data_folder_path=folder_path,
                  device="OpenEphys",
                  channels=32,
                  sampling_rate=30E3,
                  timestamps_npy="", # We can read similar to continuous.dat

                  )
#TODO: synchornized_timestamp what for shifted ??
# Masking channels for data set. Channels can be a list.
# Show user the tree. Implement representation method. filter_collection.html#FilterCollection.insert
# An example code to get the tree https://github.com/skim0119/mindinvitro/blob/master/utility/common.py
# Trimming the tree??
```

### 1.1. Meta Data Structure

```{code-cell} ipython3
# Get signal and rate(hz)
record_node: int = dataset.get_nodes[0]
recording = dataset[record_node]["experiment1"]["recording1"]   # Returns the object for recording 1
# TODO: does openephys returns the timestamp??
timestamp = recording.timestamp # returns the time stamp for the recording.

signal, _, rate = recording.continuous["100"]
# time = recording.continuous["100"].timestamp / rate
num_channels = signal.shape[1]
```

### 1.2 Raw Data

+++

If the data is provided in single `continuous.dat` instead of meta-data, user must provide number of channels and sampling rate in order to import data accurately.

> **WARNING** The size of the raw datafile can be _large_ depending on sampling rate and the amount of recorded duration. We highly recommand using meta-data structure to handle datafiles, since it only loads the data during the processing and unloads once the processing is done.

```{code-cell} ipython3
from miv.io import load_continuous_data_file

datapath = 'continuous.dat'
rate = 30_000
num_channel = 64
timestamps, signal = load_continuous_data_file(datapath, num_channel, rate)
```

## 2. Filtering Raw Signal

+++

We provide a set of basic signal filter tools. It is highly recommended to filter the signal before doing the spike-detection.
Here, we provide examples of how to create and apply the filter to the [`dataset`](../api/io.rst).

+++

If you have further suggestion on other filters to include, please leave an issue on our [GitHub issue page](https://github.com/GazzolaLab/MiV-OS/issues) with `enhancement` tag.

```{code-cell} ipython3
:tags: [hide-cell]

from miv.signal.filter import FilterCollection, ButterBandpass
```

### 2.1 Filter Collection

[Here](api/signal:signal filter) is the list of provided filters.
All filters are `Callable`, taking `signal` and `sampling_rate` as parameters.
To define a multiple filters together, we provide [`FilterCollection`](miv.signal.filter.FilterCollection) that execute multiple filters in a series.

```{code-cell} ipython3
# Butter bandpass filter
pre_filter = ButterBandpass(lowcut=300, highcut=3000, order=5)

# How to construct sequence of filters
pre_filter = (
    FilterCollection(tag="Filter Example")
        .append(ButterBandpass(lowcut=300, highcut=3000, order=5))
        #.append(Limiter(400*pq.mV))
        #.append(Filter1(**filter1_kwargs))
        #.append(Filter2(**filter2_kwargs))
)
```

### 2.2 Apply Filter

There are two way to apply the filter on the signal.
- If the signal is stored in `numpy array` format, you can directly call the filter `prefilter(signal, sampling_rate)`.
- If you want to apply the filter to all signals in the `dataset`, `dataset` provide `.apply_filter` method that takes any `filter` (any filter that abide [`filter protocol`](../api/_toctree/FilterAPI/miv.signal.filter.FilterProtocol)).
  - You can select [subset of `dataset`](miv.io.data.DataManager) and [mask-out channels](miv.io.data.Data) before applying the filter.

You can check the list of all provided filters [here](api/signal:signal filter).

```{code-cell} ipython3
# Apply filter to entire dataset
dataset.apply_filter(pre_filter)
filtered_signal = dataset[record_node]['experiment1']['recording1'].filtered_signal

# Apply filter to array
rate = 30_000
filtered_signal = pre_filter(data_array, sampling_rate=rate)

# Retrieve data from dataset and apply filter
data = dataset[record_node]['experiment1']['recording1']
filtered_signal = pre_filter(data, sampling_rate=rate)
```

## 3. Spike Detection

You can check the available method [here](api/signal:spike detection).

Most simple example of spike-detection method is using `ThresholdCutoff`.

```{code-cell} ipython3
:tags: [hide-cell]

from miv.signal.spike import ThresholdCutoff
```

```{code-cell} ipython3
# Define spike-detection method
spike_detection = ThresholdCutoff()

# The detection can be used directly as the following.
#   signal        : np.array or neo.core.AnalogSignal, shape(N_channels, N)
#   timestamps    : np.array, shape(N)
#   sampling_rate : float
timestamps = spike_detection(signal, timestamps, sampling_rate=30_000, cutoff=3.5)

# The detection can be applied on the dataset
dataset.apply_spike_detection(spike_detection)
```

## 4. Spike Visualization

```{code-cell} ipython3
:tags: [hide-cell]

import neo
from viziphant.rasterplot import rasterplot_rates
```

```{code-cell} ipython3
# Plot
rasterplot_rates(spiketrain_list)
```
