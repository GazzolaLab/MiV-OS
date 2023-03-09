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

# Connectivity using Causal Analysis

In this guide, we will explore how to use the "DirectedConnectivity" module to perform connectivity analysis and measure causal relationships among channels.

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

## Pre-Processing Pipeline

Before we dive into connectivity analysis, it's important to ensure that the data is pre-processed properly. We will create a pre-processing pipeline to filter out unwanted frequencies and detect spikes in the signal.

```{code-cell} ipython3
# Create operator modules:
bandpass_filter: Operator = ButterBandpass(lowcut=300, highcut=3000, order=4, tag="bandpass")
spike_detection: Operator = ThresholdCutoff(cutoff=4.0, use_mad=True, dead_time=0.002, tag="spikes")

data >> bandpass_filter >> spike_detection
```

## Connectivity Module

Next, we will import the "DirectedConnectivity" module from the "miv.statistics.connectivity" package. This module allows us to construct connectivity graph measured by causal relationships among channels. We will create an instance of the `DirectedConnectivity` class, and attach next to `ThresholdCutoff` operation. Two operators are compatible because the output of `ThresholdCutoff` is a `SpikeTrain` object, which is the input of `DirectedConnectivity`.

```{code-cell} ipython3
from miv.statistics.connectivity import DirectedConnectivity
connectivity_analysis = DirectedConnectivity(mea="64_intanRHD", skip_surrogate=True, progress_bar=True, channels=[11, 26, 37, 50])

spike_detection >> connectivity_analysis
print(data.summarize())  # Print the summary of data flow
```

```{note}
The "skip_surrogate" parameter is set to True to skip the computation of surrogate data. This is done to speed up the tutorial, but can be set to `False` in practice. If you are interested in learning more about surrogate data, check out the documentation.
```

```{note}
The channels parameter is set to [11, 26, 37, 50] to speed up the tutorial. In practice, you should set this parameter to `None` to analyze all channels in the network.
```

## Run Pipeline

Now that we have set up our pre-processing and connectivity analysis pipeline, we can create a pipeline object using the `Pipeline` class and run.

```{code-cell} ipython3
pipeline = Pipeline(connectivity_analysis)  # Create a pipeline object to compute connectivity
print(pipeline.summarize())
pipeline.run(working_directory="results")  # Save outcome into "results" directory
```

## Adding Centrality Plot

By default, the `DirectedConnectivity` module includes plotting the connectivity between each nodes, and plot overall connectivity matrix.
Additionally, we can visualize the centrality of the graph by adding built-in callback function, `plot_eigenvector_centrality`. Just like any other callbacks, the function can be connected to the `connectivity_analysis` operator using the `<<` operator. This will add a plot of the eigenvector centrality of the nodes in the network.

```{code-cell} ipython3
from miv.statistics.connectivity import plot_eigenvector_centrality
connectivity_analysis << plot_eigenvector_centrality
```

## Plot

Finally, we can use the "plot" method to visualize the results of our connectivity analysis. This method will generate a plot of the connectivity measures for each pair of electrodes in the network.

```{code-cell} ipython3
connectivity_analysis.plot(show=True)
```

The result would include four different plot
    1. connectivity through p-test in surrogate analysis (if skip_surrogate is true, this plot will show all nodes are connected to eachother) named `p_graph`.
    2. connectivity with connection metrics value, named `te_graph`.
    3. centrality graph of the network using eigenvector centrality, named 'eigenvector_centrality.' (This plot is added by `plot_eigenvector_centrality`)

For more information on the `DirectedConnectivity` module and other connectivity analysis tools in the `miv` package, check out the API documentation.
