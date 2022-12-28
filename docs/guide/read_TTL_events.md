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

```{code-cell} ipython3
:tags: [hide-cell]

from miv.datasets.ttl_events import load_data
```

## Load example dataset

TTL event are compose of five data types: states, full_words, timestamps, sampling rate, and initial state.

- The details of what each data represent is described [here](https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html#events).
- The API documentation is [here](file:///Users/skim0119/github/MiV-OS/docs/_build/html/api/io.html#miv.io.binary.apply_channel_mask).

> Note: The timestamps data are already synchronized with other recording streams. To match the time, make sure to turn off `start_at_zero` when loading the signal.

```{code-cell} ipython3
:tags: [hide-cell]

datasets = load_data()
```

```{code-cell} ipython3
data = datasets[0]
states, full_words, timestamps, sampling_rate, initial_state = data.load_ttl_event()
```

## Visualize TTL Events

```{code-cell} ipython3
:tags: [hide-cell]

import numpy as np
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
on = timestamps[states == 1]
off = timestamps[states == -1]

for start, end in zip(on, off):
    plt.axvspan(start, end, alpha=0.4, color='red')
plt.title("TTL ON/OFF Events")
plt.xlabel("time (s)")
```
