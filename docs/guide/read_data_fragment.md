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
  execution_mode: 'off'
---

# Handling Large Datafile

## Prepare Spike Detection

```{code-cell} ipython3
import os, sys
import numpy as np

from tqdm import tqdm

from miv.io import Data, DataManager
from miv.signal.filter import ButterBandpass
from miv.signal.spike import ThresholdCutoff

from miv.core import Spikestamps

pre_filter = FilterCollection(tag="Filter Example").append(
    ButterBandpass(lowcut=300, highcut=3000, order=4)
)
spike_detection = ThresholdCutoff()

def preprocess(data, filter:FilterProtocol, detector:SpikeDetectionProtocol):
    signal, timestamps, sampling_rate = data
    timestamps *= sampling_rate  # Still not sure about this part...
    filtered_signal = filter(signal, sampling_rate)
    spiketrains = detector(
        filtered_signal,
        timestamps,
        sampling_rate,
        return_neotype=False,
        progress_bar=False,
    )
    return spiketrains
```

## Read Data Fragmentally

Instead of `data.load()`, try `data.load_fragments(num_fragments)` which internally splits the large datafile into
number of fragmented data.

```{code-cell} ipython3
num_fragments = 100
total_spikestamps = Spikestamps([])
for data in tqdm(data.load_fragments(num_fragments=num_fragments), total=num_fragments):
    spikestamp = preprocess(data, pre_filter, spike_detection)
    total_spikestamps.extend(spikestamp)
```

## Using Multiprocessing

You can use multiprocessing module as following.

```{code-cell} ipython3
import multiprocessing as mp

num_fragments = 100
total_spikestamps = Spikestamps([])
with mp.Pool() as pool:
    results = list(
        tqdm(
            pool.imap(
                partial(preprocess, filter=pre_filter, detector=spike_detection),
                data.load_fragments(num_fragments=num_fragments),
            ),
            total=num_fragments,
        )
    )
    for spikestamp in results:
        total_spikestamps.extend(spikestamp)
```

## Using MPI

## Benchmark: Optimal num_fragments
