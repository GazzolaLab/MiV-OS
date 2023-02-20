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
  execution_mode: 'cache'
---

# Introduction : Quick Start

Here is a quick-start example of how to build electrophysiology analysis pipelines using `MiV-OS`.

```{code-cell} ipython3
:tags: [hide-cell]
import numpy as np
import quantities as pq

from miv.core.operator import Operator, DataLoader
from miv.core.pipeline import Pipeline
from miv.io.openephys import Data, DataManager
from miv.signal.filter import ButterBandpass, MedianFilter
from miv.signal.spike import ThresholdCutoff

from miv.datasets.openephys_sample import load_data
```

We will be using the recording from OpenEphys data aquisition system.

```{code-cell}
# Download the sample data
path:str = load_data(progbar_disable=True).data_collection_path
print(path)
```

## Analysis Pipeline

The analysis starts by creating operation modules.

```{code-cell} ipython3
# Create data modules:
dataset: DataManager = DataManager(data_collection_path=path)
data: DataLoader = dataset[0]

# Create operator modules:
bandpass_filter: Operator = ButterBandpass(lowcut=300, highcut=3000, order=4)
lfp_filter: Operator = ButterBandpass(highcut=3000, order=2, btype='lowpass')
spike_detection: Operator = ThresholdCutoff(cutoff=4.0, use_mad=True, dead_time=0.002)
```

Next, the chain of analysis pipeline can be built using `>>` operator.
The order of operation does not necessarily needs to be in single sequence.
Following example pass recorded signal into bandpass filter and LFP filter, and pass bandpass filtered data into threshold-based spike detection.

```{code-cell} ipython3
# Build analysis pipeline
data >> bandpass_filter >> spike_detection
data >> lfp_filter
```

To visualize the flow structure, we can use `.summarize()` method starting from any operator:

```{code-cell} ipython3
print(data.summarize())
```

To finalize the analysis, we can create a `Pipeline` object and run.

```{code-cell} ipython3
pipeline = Pipeline(data)
pipeline.run(save_path="results/")  # Save outcome into "results" directory
```

```{note}
Internally the data is splitted into 1 minute fragments to process by default. To change the value, check API documentation (here)[miv.io.protocol.DataProtocol].
```

To access the result from specific module, you can query using `.output` property.
For some operations with large output result, the query might returns a generator for fragmented result.

```{code-cell} ipython3
filtered_signal = bandpass_filter.output
```

```{note}
The OpenEphys DataManager provides a method `dataset.tree()` to check the recording.
```


```{note}
We provide a set of basic signal filter tools.
You can check the list of all provided filters [here](../api/signal).
```

### Filter Collection

This allows us to easily construct a sequence of filters:

```{code-cell} ipython3
median_filter = MedianFilter(threshold=100 * pq.mV)
data >> median_filter >> bandpass_filter
pipeline = Pipeline(data)
```

### Visualize Pipeline

We provide `pipeline.summarize()` method to print the order of processing operations.

```{code-cell} ipython3
print(pipeline.summarize())
```

## Rasterplot Visualization

```{code-cell} ipython3
# Plot
spike_detection.plot(show=True)
```
