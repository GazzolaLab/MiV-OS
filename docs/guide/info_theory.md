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

# Information Theory Module

Information theory techniques are being increasingly employed in neuroscience to understand the underlying connectivity and dynamics.[1][1] Current module contains some commonly employed measures applied to binned spiketrains. [2][2]

References:

[1]: https://www.eneuro.org/content/5/3/ENEURO.0052-18.2018
[2]: https://elife-asu.github.io/PyInform/starting.html

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

## 2. Shannon Entropy
Calculates Shannon entropy for the specified channel. Documentation is available [here](miv.statistics.info_theory.shannon_entropy).

```{code-cell} ipython3
# Estimates the shannon entropy for channel number 20, with signal taken from 0 to 120 seconds and bin interval of 2ms
sh_entropy(spiketrains, 20, 0, 120, 0.002)
```

## 3. Block Entropy
Calculates Block entropy for the specified channel. Documentation is available [here](miv.statistics.info_theory.block_entropy).

```{code-cell} ipython3
# Estimates the block entropy for channel number 20, with signal taken from 0 to 120 seconds and bin interval of 2ms. The history length is 1
block_entropy(spiketrains, 20, 1, 0, 120, 0.002)
```

## 4. Entropy Rate
Calculates Entropy rate for the specified channel. Documentation is available [here](miv.statistics.info_theory.entropy_rate).

```{code-cell} ipython3
# Estimates the entropy rate for channel number 20, with signal taken from 0 to 120 seconds and bin interval of 2ms. The history length is 1
entropy_rate(spiketrains, 20, 1, 0, 120, 0.002)
```

## 5. Active Information
Calculates active information for the specified channel. Documentation is available [here](miv.statistics.info_theory.active_info).

```{code-cell} ipython3
# Estimates the active information for channel number 20, with signal taken from 0 to 120 seconds and bin interval of 2ms. The history length is 1
active_info(spiketrains, 20, 1, 0, 120, 0.002)
```

## 6. Mutual Information
Estimates the mutual information for the pair of electorde recordings (X & Y). Documentation is available [here](miv.statistics.info_theory.mutual_info).

```{code-cell} ipython3
# Estimates the mutual information for channel number 20 and 21, with signal taken from 0 to 120 seconds and bin interval of 2ms.
mutual_info(spiketrains, 20, 21, 0, 120, 0.002)
```

## 7. Relative Entropy
Estimates the relative entropy for the pair of electorde recordings (X & Y) . Documentation is available [here](miv.statistics.info_theory.relative_entropy).

```{code-cell} ipython3
# Estimates the relative entropy for channel number 20 and 21, with signal taken from 0 to 120 seconds and bin interval of 2ms.
relative_entropy(spiketrains, 20, 21, 0, 120, 0.002)
```

## 8. Conditional Entropy
Estimates the conditional entropy for the pair of electorde recordings (X & Y) . Documentation is available [here](miv.statistics.info_theory.conditional_entropy).

```{code-cell} ipython3
# Estimates the conditional entropy for channel number 20 and 21, with signal taken from 0 to 120 seconds and bin interval of 2ms.
conditional_entropy(spiketrains, 20, 21, 0, 120, 0.002)
```

## 8. Transfer Entropy
Estimates the transfer entropy for the pair of electorde recordings (X & Y) . Documentation is available [here](miv.statistics.info_theory.transfer_entropy).

```{code-cell} ipython3
# Estimates the transfer entropy for channel number 20 and 21, with signal taken from 0 to 120 seconds and bin interval of 2ms. The history length is 1
transfer_entropy(spiketrains, 20, 21, 1, 0, 120, 0.002)
```
