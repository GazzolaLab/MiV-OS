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
  execution_mode: 'cache'
---

# Spike Cutout Visualization

```{code-cell} ipython3
:tags: [hide-cell]

import numpy as np
import quantities as pq
import matplotlib.pyplot as plt

from miv.core.operator import Operator, DataLoader
from miv.core.pipeline import Pipeline
from miv.io.openephys import Data, DataManager
from miv.signal.filter import ButterBandpass, MedianFilter
from miv.signal.spike import ThresholdCutoff

from miv.datasets.openephys_sample import load_data

# Prepare data
dataset: DataManager = load_data(progbar_disable=True)
data: DataLoader = dataset[0]
```

## Build Pipeline

We will first create a pre-processing pipeline.
We use `bandpass_filter` and the `spike_detection`, using the `ButterBandpass` and `ThresholdCutoff` classes, respectively.

```{code-cell} ipython3
# Create operator modules:
bandpass_filter: Operator = ButterBandpass(lowcut=300, highcut=3000, order=4, tag="bandpass")
bandpass_filter2: Operator = ButterBandpass(lowcut=300, highcut=3000, order=4, tag="bandpass2")
spike_detection: Operator = ThresholdCutoff(cutoff=4.0, use_mad=True, dead_time=0.002, tag="spikes")
```

We can assign a `tag` to each operator to label it and help visualize the processing steps later.

## Configure Spike Cutout Module

Next, we introduce the `ExtractWaveforms` module and configure it to extract spikes from the filtered signal.
This operator extracts the waveforms of spikes in a signal and aligns them by their peak.
It requires a list of channels to extract the spikes from, which we can specify using the `channels` parameter. (To extract spikes from all channels, we can set `channels=None`.)
We can also limit the number of spikes to plot using the `plot_n_spikes` parameter.

```{code-cell} ipython3
from miv.signal.events import ExtractWaveforms

extract_waveforms: Operator = ExtractWaveforms(channels=[11, 26, 37, 50], plot_n_spikes=150)
```

With these operators created, we can now build our pipeline by chaining them together using the `>>` operator.
We will start with the `bandpass_filter`, followed by the `spike_detection`, and finally the `extract_waveforms` operator.

```{code-cell} ipython3
data >> bandpass_filter2 >> spike_detection
data >> bandpass_filter
bandpass_filter >> extract_waveforms
spike_detection >> extract_waveforms
```

This creates a processing chain that filters the recorded signal, detects spikes in the filtered signal, and then extracts the waveforms of these spikes.
Notice, `extract_waveform` module takes two inputs: `Signal` and `Spikestamps`, which is why the module takes two inputs, from `bandpass_filter` and `spike_detection`, respectively.
**The order of the inputs is important** when defining the processing chain.

Before proceeding further, let's visualize the data pipeline to ensure the processing order.

```{code-cell} ipython3
data.visualize(show=True)
```

## Run Pipeline

We can then create a `Pipeline` object and run it to execute the processing steps.

```{code-cell} ipython3
pipeline = Pipeline(extract_waveforms)
print(pipeline.summarize())

pipeline.run(verbose=True)
```

## Plot Spike Cutouts

Just like the `spike_detection` module, the `extract_waveforms` module also has a `plot` method that can be used to visualize the extracted spike cutouts.

```{code-cell} ipython3
extract_waveforms.plot(show=True)
```

This will display a plot of the extracted spike cutouts. We can also access the extracted cutouts using the `output` property of the `extract_waveforms` module:

```{code-cell} ipython3
cutouts = extract_waveforms.output
for ch, cutout in cutouts.items():
    print(f"Channel {ch}: {cutout.data.shape}")
```
