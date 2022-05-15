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
---

# Data Management

- Data
- DataManager
- load_continuous_data (raw)

```{code-cell} ipython3
:tags: [hide-cell]

import os
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from glob import glob
from miv.io import *
```

```{code-cell} ipython3
datapath = './2022-03-10_16-19-09'
os.path.exists(datapath)
```

```{code-cell} ipython3
filepath = './2022-03-10_16-19-09/Record Node 104/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat'
os.path.exists(filepath)
```

## 1. Data Load

```{code-cell} ipython3
# Load dataset from OpenEphys recording
folder_path: str = "~/Open Ephys/2022-03-10-16-19-09"  # Data Path
# Provide the path of experimental recording tree to the DataSet class
# Data set class will load the data and create a list of objects for each data
# dataset = load_data(folder_path, device="OpenEphys")
dataset = Dataset(data_folder_path=folder_path,
                  device="OpenEphys",
                  channels=32,
                  sampling_rate=30E3,
                  timestamps_npy="", # We can read similar to continuous.dat

                  )
#TODO: synchornized_timestamp what for shifted ??
# Masking channels for data set. Channels can be a list.
# Show user the tree. Implement representation method. filter_collection.html#FilterCollection.insert
# An example code to get the tree https://github.com/skim0119/mindinvitro/blob/master/utility/common.py
# Trimming the tree??
```

### 1.1. Meta Data Structure

```{code-cell} ipython3
# Get signal and rate(hz)
record_node: int = dataset.get_nodes[0]
recording = dataset[record_node]["experiment1"]["recording1"]   # Returns the object for recording 1
# TODO: does openephys returns the timestamp??
timestamp = recording.timestamp # returns the time stamp for the recording.

signal, _, rate = recording.continuous["100"]
# time = recording.continuous["100"].timestamp / rate
num_channels = signal.shape[1]
```

### 1.2 Raw Data

+++

If the data is provided in single `continuous.dat` instead of meta-data, user must provide number of channels and sampling rate in order to import data accurately.

> **WARNING** The size of the raw datafile can be _large_ depending on sampling rate and the amount of recorded duration. We highly recommand using meta-data structure to handle datafiles, since it only loads the data during the processing and unloads once the processing is done.

```{code-cell} ipython3
from miv.io import load_continuous_data_file

datapath = 'continuous.dat'
rate = 30_000
num_channel = 64
timestamps, signal = load_continuous_data_file(datapath, num_channel, rate)
```

## 2. Instant Visualization
