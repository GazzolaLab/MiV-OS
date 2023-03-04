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

# Callback: Customized Plotting

In this tutorial, we will learn how to embedd customized plotting script into the operator.

```{code-cell} ipython3
:tags: [hide-cell]

import numpy as np
import matplotlib.pyplot as plt

from miv.core.datatype import Signal
from miv.core.operator import Operator, DataLoader
from miv.core.pipeline import Pipeline
from miv.io.openephys import DataManager
from miv.signal.filter import ButterBandpass

from miv.datasets.openephys_sample import load_data

# Prepare data
dataset: DataManager = load_data(progbar_disable=True)
data: DataLoader = dataset[0]
```

## What is Callback

A callback is a function that is called at a specific point during the operator.
The callback function can be used to perform any custom operation, such as plotting, saving, or logging.
In `MiV-OS`, we provide a way to embedd customized callback script during post-processing stage of the operator.

```{note}
The callback for pre-processing stage is not yet supported. It will be included in the future.
```

## How to Inject Custom Callback

Take the bandpass filter as an example, we can define a callback function to plot the filtered signal.

```{code-cell} ipython3
bandpass_filter: Operator = ButterBandpass(lowcut=300, highcut=3000, order=4, tag="bandpass")
data >> bandpass_filter
```

The callback function should be defined as a function with the following signature: `def callback(self, signal: Signal): ...`.
The parameter of the function, `signal` argument in this case, must match the output of the operator.
In this example, the `signal` argument is the output of the `bandpass_filter` operator, which is a `Signal` object.

> Note: The `self` argument is the operator instance itself. It is required to access the operator's properties, such as `self.analysis_path`.

```{code-cell} ipython3
def callback_statistics(self, signal:Signal):
    """Print the statistics of the filtered signal"""
    signal = next(signal)  # Get the first signal fragment
    for channel in range(5):
        print(f"{channel=} | mean={signal.data[channel].mean():.2f} | std={signal.data[channel].std():.2f} | median={np.median(signal.data[channel]):.2f}")
    return signal

def callback_median_histogram(self, signal:Signal):
    """Plot the histogram of the median of each channel"""
    medians = []
    for channel in range(signal.number_of_channels):
        medians.append(np.median(signal.data[channel]))
    plt.hist(medians, bins=20)
    plt.title("Histogram of the median of each channel")
    plt.xlabel("Median (mV)")
    plt.ylabel("Count")
    return signal
```

We can then pass the callback function to the operator using `<<` operator.

```{code-cell} ipython3
bandpass_filter << callback_statistics << callback_median_histogram
```

```{note}
The callback function will be called __in the order__ of the `<<` operator. In the above example, the `callback_statistics` function will be called first, followed by `callback_median_histogram`, hence the input of the `callback_median_histogram` function is the output of the `callback_statistics` function.
```

Lets run the pipeline and see the result.

```{code-cell} ipython3
pipeline = Pipeline(bandpass_filter)
pipeline.run()
```

As we can see, the callback function is called after the operator is executed.
