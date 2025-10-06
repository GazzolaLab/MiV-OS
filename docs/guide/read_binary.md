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

# Read Raw Binary File

Some interface and recording device saves in raw binary format. Here, we provide a method of how to import raw binary signal.

We assume the data is saved in column-major matrix, where the shape of the matrix is (num_sample, num_channel).
For example, the binary with N sample M channel is saved in the order (sample 1 from channel 1), (sample 1 from sample 2), ..., (sample 1 of channel N), (sample 2 of channel 1), ..., (sample N of channel M).

> This method can be used to load large data sample that cannot fit in RAM memory at once. In such case, be careful to process data in segment.
> When reading the data directly from binary file, it is user's responsibility to convert binary data to voltage.

```{code-cell} ipython3
:tags: [hide-cell]

import os, sys
import numpy as np

from tqdm import tqdm

from miv.io import load_continuous_data
from miv.signal.filter import ButterBandpass
from miv.signal.spike import ThresholdCutoff
from miv.core import Spikestamps
```

## Import Data

Since raw data does not contain the meta-data of recording, user must specify number of channels and sampling rate.

```{code-cell} ipython3
file_path: str    = "data_16bit.bin"  # Path to binary file
num_channels: int = 512
sampling_rate:int = 20_000
```

To read the binary file, use `load_continuous_data` function from `miv.io`.

```{code-cell} ipython3
raw_signal, timestamps = load_continuous_data(file_path, num_channels=num_channels, sampling_rate=sampling_rate)
print(raw_signal.shape)  # (num_samples, num_channels)
print(timestamps.shape)  # (num_samples)
```

## Prepare Spike Detection

For demonstration, lets use a simple bandpass filter with threshold detection to get spikestamps.

```{code-cell} ipython3
pre_filter = ButterBandpass(lowcut=300, highcut=3000, order=5)
spike_detection = ThresholdCutoff()
```

## Get SpikeTrain

To iterate segment of large dataset, it is recommended to use `np.array_split` which returns array `view` of partial segment.

```{code-cell} ipython3
total_spiketrain = Spikestamps([])

# Split data into fragment of 1 minute recordings
n_split = timestamps.shape[0] // sampling_rate // 60  # Split for each 1 minute (60 seconds) of data
raw_signal_split = np.array_split(raw_signal, n_split, axis=0)
timestamps_split = np.array_split(timestamps, n_split)

# Process
for signal_seg, timestamps_seg in tqdm(zip(raw_signal_split, timestamps_split), total=n_split):
    filtered_signal = pre_filter(signal_seg, sampling_rate=sampling_rate)
    spks = spike_detection(filtered_signal, timestamps_seg, sampling_rate=sampling_rate, progress_bar=False)
    total_spiketrain.extend(spks)

print(total_spiketrain.get_count())
```
