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

# Change Point Detection

Change point detection (CPD) is an algorithmic method to identify the moments when the statistical properties of a data sequency changes.
Given the temporal sequence of data with N number of features, algorithm tries to find the set of change-points, such that the best piece-wise regression loss (using basis of RBF, polynomial, sinusoidal, etc.) is minimized.
Using this method effectively can help summarize the data by highlighting important changes in the underlying trends and patterns. [1][1]

Here, we provide example of using the Python package [ruptures](https://centre-borelli.github.io/ruptures-docs/), which provides several CPD algorithms that can directly be applied to neural recordings, using moving averaged firing rate.

[1]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5464762/#:~:text=Change%20point%20detection%20(CPD)%20is,well%20as%20change%20point%20detection.

+++

## Load Modules

```{code-cell} ipython3
import miv
from miv.statistics import firing_rates
from miv.core import Spikestamps
from miv.statistics import binned_spiketrain

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

import ruptures as rpt
```

### Moving Average Functions

```{code-cell} ipython3
def moving_average(a, n=3) :
    return np.convolve(a, np.ones(n), 'valid') / n
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
```

## Get Moving Averaged Firing Rate

We start with pre-processed data in `Spikestamps` form (To preprocess, see [here](../tutorial/signal_processing.md)).

```{code-cell} ipython3
spiketrain = Spikestamps(...)  # from pre-processing
duration = spiketrain.get_last_spikestamp() - spiketrain.get_first_spikestamp()
```

```{code-cell} ipython3
bin_size = 1.0        # sec
moving_window = 60    # sec
minimum_spikerate = 1 # spikes / sec

moving_averaged_firing_rate = []
for idx, spk in enumerate(spiketrain):
    if len(spk) / (duration) < minimum_spikerate:
        continue

    bspk = binned_spiketrain(
        spk,
        spiketrain.get_first_spikestamp(),
        spiketrain.get_last_spikestamp(),
        bin_size,
        return_count=True
    )
    ma = moving_average(bspk, n=ma_window)
    moving_averaged_firing_rate.append(ma)

moving_averaged_firing_rate = np.asarray(moving_averaged_firing_rate)
print(moving_averaged_firing_rate.shape)
```

## Ruptures - Dynamic Programming

```{code-cell} ipython3
model = "l2"  # "l2", "rbf"
algo = rpt.Dynp(model=model, min_size=3, jump=600).fit(moving_averaged_firing_rate.T)
```

```{code-cell} ipython3
n_points = 10
my_bkps = algo.predict(n_points)
```

```{code-cell} ipython3
# show results
fig, ax_arr = rpt.display(moving_averaged_firing_rate.T, my_bkps)
plt.show()
```

## Ruptures - Kernel-Change

```{code-cell} ipython3
algo = rpt.KernelCPD(kernel='linear', min_size=3, jump=600).fit(moving_averaged_firing_rate.T)
```

```{code-cell} ipython3
penalty = 1000
my_bkps = algo.predict(pen=penalty)
print(len(my_bkps))
```

```{code-cell} ipython3
# show results
fig, ax_arr = rpt.display(moving_averaged_firing_rate.T, my_bkps)
plt.show()
```
