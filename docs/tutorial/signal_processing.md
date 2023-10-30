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

# Introduction: `MiV-OS` Pipeline

## Electrophysiology Analysis Pipelines

In this tutorial, I will provide you with a quick-start example that demonstrates how to construct electrophysiology analysis pipelines utilizing `MiV-OS`. By the end of this tutorial, you will have gained a better understanding of how to apply this package to your own electrophysiology experiments.

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
```

For the purpose of this tutorial, we will be using the recording data obtained from the OpenEphys data acquisition system.
By running `miv.datasets.openephys_sample.load_data()`, we can download the sample data.

```{code-cell}
# Download the sample data
path:str = load_data(progbar_disable=True).data_collection_path
print(path)
```

## Analysis Pipeline

To commence the analysis, the first step is to create operation modules, which will be used to perform specific tasks on the data.
The following code block demonstrates how to create data and operator modules:

```{code-cell} ipython3
# Create data modules:
dataset: DataManager = DataManager(data_collection_path=path)
data: DataLoader = dataset[0]

# Create operator modules:
bandpass_filter: Operator = ButterBandpass(lowcut=300, highcut=3000, order=4)
lfp_filter: Operator = ButterBandpass(highcut=3000, order=2, btype='lowpass')
spike_detection: Operator = ThresholdCutoff(cutoff=4.0, dead_time=0.002)
```

In this code block, we are creating three different operator modules, namely `bandpass_filter`, `lfp_filter`, and `spike_detection`.
These modules will be used later in the pipeline to filter and detect spikes in the electrophysiological data.
Additionally, we are creating a `DataManager` module to load the data from the specified path, and a `DataLoader` module to extract specfiic experimental data from the `DataManager` module.

```{note}
The OpenEphys DataManager provides a method `dataset.tree()` to check the recording.
```

Once the operation modules have been created, the analysis pipeline can be constructed using the `>>` operator.
The order of operations in the pipeline does not have to be a serial sequence.
For instance, the following example passes the recorded signal through the bandpass filter and LFP filter, and then passes the bandpass filtered data through the threshold-based spike detection module:

```{code-cell} ipython3
# Build analysis pipeline
data >> bandpass_filter >> spike_detection
data >> lfp_filter

print(data.summarize())  # Print the summary of data flow
```

To complete the analysis, we can create a `Pipeline` object and execute it by calling the `.run()` method.
This will run the operations specified in the pipeline and generate the desired outcomes.
The results can be saved to a specified directory by passing the path to the `working_directory` argument.

```{code-cell} ipython3
pipeline = Pipeline(spike_detection)  # Create a pipeline object to get `spike_detection` output
pipeline.run(working_directory="results/")  # Save outcome into "results" directory
```

```{note}
Internally the data is splitted into 1 minute fragments to process by default. To change the value, check API documentation (here)[miv.io.protocol.DataProtocol].
```

### Querying Results

To access the result from specific module, you can query using `.output` property.
For some operations with large output result, the query might returns a generator for fragmented result.

To retrieve the output result from a specific module in the pipeline, you can use the `.output` property followed by the name of the module.
If the output result is large, the query might return a `Python generator` for fragmented result.
The following example demonstrates how to retrieve the output from the bandpass_filter module:

```{code-cell} ipython3
filtered_signal = next(bandpass_filter.output())  # Next is used to retrieve the first fragment of the output
print(filtered_signal.shape)

time = filtered_signal.timestamps
elctrode = filtered_signal.data[:, 11]

plt.figure(figsize=(10, 5))
plt.plot(time, elctrode)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.show()
```

You can then use the `filtered_signal` variable to perform additional analysis or to visualize the filtered signal.

### Filter Collection

With `MiV-OS`, it is straightforward to construct a sequence of filters by chaining the operator modules.
For instance, the following code block shows how to construct a median filter followed by a bandpass filter in a single sequence:

```{code-cell} ipython3
median_filter = MedianFilter(threshold=100 * pq.mV)
data >> median_filter >> bandpass_filter
pipeline = Pipeline(bandpass_filter)
```

In this example, we first create a `MedianFilter` module with a specified threshold of 100 mV, and then pass the output from this module into the `bandpass_filter` module.
The resulting pipeline can then be executed using a `Pipeline` object, as described earlier.

### Visualize Pipeline

To view the order of processing operations in a pipeline, we can use the `summarize()` method of a `Pipeline` object.
This method construct a summary of the pipeline in text form, including the sequence of operations that will be applied to the input data.
The following code block demonstrates how to use the `summarize()` method:

```{code-cell} ipython3
print(pipeline.summarize())
```

By calling this method, you can quickly confirm the sequence of operations in the pipeline and ensure that it matches your intended workflow.

## Rasterplot Visualization

The visualization of an operation in `MiV-OS` is attached to each modules, and it is executed at the end of running each block modules.
For example, the `spike_detection` module have one plotting method that visualize the detected spikes in a rasterplot.
The method `.plot()` will call _every_ plotting methods in the module, and save the result in the working directory.
