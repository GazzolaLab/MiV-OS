---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
file_format: mystnb
mystnb:
  execution_mode: 'off'
---

# Introduction : Quick Start

Here is a quick-start example of how to start using `MiV-OS`.

```{code-cell} ipython3
:tags: [hide-cell]

import os
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
```

## 1. Data Load (Open Ephys)

We natively support the output file-structure that `OpenEphys` uses.

```{code-cell} ipython3
:tags: [hide-cell]

from miv.io.data import Data, DataManager
```

```{code-cell} ipython3
# Load dataset from OpenEphys recording
folder_path: str = "2022-03-10_16-19-09/"  # Data Path
# Provide the path of experimental recording tree to the DataManager class
dataset = DataManager(folder_path)
```

```{code-cell} ipython3
dataset.tree()
```

You should be able to check the data structure by running `dataset.tree()`.


## 2. Filtering Raw Signal

+++

We provide a set of basic signal filter tools.
Here, we provide examples of how to create and apply the filter to the [`dataset`](../api/io.rst).

+++

If you have further suggestion on other filters to include, please leave an issue on our [GitHub issue page](https://github.com/GazzolaLab/MiV-OS/issues) with `enhancement` tag.

```{code-cell} ipython3
:tags: [hide-cell]

from miv.signal.filter import FilterCollection, ButterBandpass
```

### 2.1 Filter Collection

All filters are directly `Callable`, taking `signal` and `sampling_rate` as parameters.
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

You can check the list of all provided filters [here](../api/signal).

```{code-cell} ipython3
# Apply filter to `dataset[0]`
with dataset[0].load() as (signal, timestamps, sampling_rate):
    filtered_signal = pre_filter(signal, sampling_rate)
```

## 3. Spike Detection

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

# The detection can be applied on the dataset
spiketrains = spike_detection(filtered_signal, timestamps, sampling_rate)
```

## 4. Spike Visualization

```{code-cell} ipython3
:tags: [hide-cell]

import neo
from viziphant.rasterplot import rasterplot_rates
```

```{code-cell} ipython3
# Plot
rasterplot_rates(spiketrains)
```
