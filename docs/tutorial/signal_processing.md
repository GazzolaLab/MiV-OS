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
We demonstrate how to build analysis pipeline for recorded electrophysiology data.

```{code-cell} ipython3
import numpy as np
import quantities import pq

from miv.io.openephys import Data, DataManager
from miv.signal.filter import ButterBandpass, MedianFilter
from miv.signal.spike import ThresholdCutoff

import neo
from viziphant.rasterplot import rasterplot_rates
```

## Analysis Pipeline

The analysis starts by creating operation modules.

```{code-cell} ipython3
# Create operation modules:
dataset = DataManager(data_collection_path="2022-03-10_16-19-09")
bandpass_filter = ButterBandpass(lowcut=300, highcut=3000, order=4)
lfp_filter = ButterBandpass(highcut=3000, order=2, btype='lowpass')
spike_detection = ThresholdCutoff(cutoff=4.0, use_mad=True, dead_time=0.002)
```

Next, the chain of analysis pipeline can be built using `>>` operator.
The order of operation does not necessarily needs to be in single sequence.
Following example pass recorded signal into bandpass filter and LFP filter, and pass bandpass filtered data into threshold-based spike detection.

```{code-cell} ipython3
# Build analysis pipeline
pipeline = dataset >> bandpass_filter >> spike_detection
dataset >> lfp_filter
pipeline.run()
pipeline.export("results/")  # Save outcome into "results" directory
```

```{note}
Internally the data is splitted into 1 minute fragments to process by default. To change the value, check API documentation (here)[miv.io.protocol.DataProtocol].
```

To access the result from specific module, you can query using `pipeline.get` method.
For some operations with large output result, the query might returns fragmented result.

```{code-cell} ipython3
filtered_signal = pipeline.query(bandpass_filter)
```

```{note}
The OpenEphys DataManager provides a method `dataset.tree()` to check the recording.
```


```{note}
We provide a set of basic signal filter tools.
You can check the list of all provided filters [here](../api/signal).
```

### Filter Collection

Such structure allow us to easily create sequence of filters.
To define a multiple filters together try:

```{code-cell} ipython3
median_filter = MedianFilter(threshold=100 * pq.mV)
filter_set = median_filter >> bandpass_filter
pipeline = dataset >> filter_set
```

### Visualize Pipeline

We provide `pipeline.summarize()` and `pipeline.visualize()` method to illustrate the processing pipeline.
The method `summarize()` returns string of pipeline summary in text format, and `visualize()` returns graphical figure (`matplotlib.pyplot.Figure`).

```{code-cell} ipython3
pipeline.summary()
```

```{code-cell} ipython3
pipeline.visualize()
plt.show()
```

## Rasterplot Visualization

```{code-cell} ipython3
# Plot
spiketrains = pipeline.query(spike_detection)
rasterplot_rates(spiketrains)
```
