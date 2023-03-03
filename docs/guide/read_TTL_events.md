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
---

# Read TTL Events in OpenEphys Data

This guide includes steps of how to load TTL events from OpenEphys binary data.

> Use the following tutorial-data to run the code. If you have your own recorded data, you can directly load it using `DataManager`.

```{code-cell} ipython3
:tags: [hide-cell]

from miv.datasets.ttl_events import load_data
datasets = load_data()  # sample DataManager with TTL recording
data = datasets[0]
```

## Load example dataset

TTL event are compose of five data types: states, full_words, timestamps, sampling rate, and initial state.

- The details of what each data represent is described [here](https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html#events).
- The API documentation is [here](file:///Users/skim0119/github/MiV-OS/docs/_build/html/api/io.html#miv.io.binary.apply_channel_mask).

> Note: The timestamps data are already synchronized with other recording streams. To match the time, make sure to turn off `start_at_zero` when loading the signal.

```{code-cell} ipython3
data = datasets[0]
signal = data.load_ttl_event()

states = signal.data[:,0]
timestamps = signal.timestamps
sampling_rate = signal.rate
```

## Visualize TTL Events

Here is example script to visualize TTL event.

```{code-cell} ipython3
:tags: [hide-cell]

import numpy as np
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
ttl_port = 1
on = timestamps[states == ttl_port]
off = timestamps[states == -ttl_port]

for start, end in zip(on, off):
    plt.axvspan(start, end, alpha=0.4, color='red')
plt.title("TTL ON/OFF Events")
plt.xlabel("time (s)")
```
