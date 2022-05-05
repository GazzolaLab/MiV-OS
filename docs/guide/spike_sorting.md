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

#

## References

- Spike sorting based on discrete wavelet transform coefficients (Letelier 2000)
- Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering (Quiroga 2004)
- A novel and fully automatic spike-sorting implementation with variable number of features (Chaure 2018)

```{code-cell} ipython3
:tags: [hide-cell]

import os, sys
import numpy as np
import scipy
import scipy.special
import quantities as pq
import matplotlib.pyplot as plt
import pywt
```

```{code-cell} ipython3
:tags: [remove-cell]

sys.path.append('../..')
```

```{code-cell} ipython3
:tags: [remove-cell]

from miv.io import load_continuous_data_file

datapath = '2022-03-10_16-19-09/Record Node 104/spontaneous/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat'
rate = 30_000
timestamps, signal = load_continuous_data_file(datapath, 64, rate)
```

## Pre-Filter

```{code-cell} ipython3
:tags: [hide-cell]

from miv.signal.filter import ButterBandpass
from miv.signal.spike import ThresholdCutoff
```

```{code-cell} ipython3
pre_filter = ButterBandpass(lowcut=300, highcut=3000, order=5)
filtered_signal = pre_filter(signal, sampling_rate=rate)

spike_detection = ThresholdCutoff()
spks = spike_detection(filtered_signal, timestamps, sampling_rate=30_000, progress_bar=False)
```

## Plot

```{code-cell} ipython3
:tags: [hide-cell]

from miv.visualization import extract_waveforms, plot_waveforms
```

```{code-cell} ipython3
cutouts = extract_waveforms(
    filtered_signal, spks, channel=7, sampling_rate=rate
)
plot_waveforms(cutouts, rate, n_spikes=250)
```

## Simple Clustering

```{code-cell} ipython3
:tags: [hide-cell]

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

```{code-cell} ipython3
scaler = StandardScaler()
scaled_cutouts = scaler.fit_transform(cutouts)

pca = PCA()
pca.fit(scaled_cutouts)
# print(pca.explained_variance_ratio_)

pca.n_components = 2
transformed = pca.fit_transform(scaled_cutouts)
```
```{code-cell} ipython3
# Clustering
n_components = 3 # Number of clustering components
gmm = GaussianMixture(n_components=n_components, n_init=10)
labels = gmm.fit_predict(transformed)
```

```{code-cell} ipython3
tmp_list = []
for i in range(n_components):
    idx = labels == i
    tmp_list.append(timestamps[idx])
    spikestamps_clustered.append(tmp_list)

_ = plt.figure(figsize=(8, 8))
for i in range(n_components):
    idx = labels == i
    _ = plt.plot(transformed[idx, 0], transformed[idx, 1], ".")
    _ = plt.title("Cluster assignments by a GMM")
    _ = plt.xlabel("Principal Component 1")
    _ = plt.ylabel("Principal Component 2")
    _ = plt.legend([0, 1, 2])
    _ = plt.axis("tight")

_ = plt.figure(figsize=(8, 8))
for i in range(n_components):
    idx = labels == i
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
    plot_waveforms(
        cutouts[idx, :], rate, n_spikes=100, color=color,
    )
# custom legend
custom_lines = [plt.Line2D([0], [0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"][i], lw=4,) \
                    for i in range(n_components)]
plt.legend(custom_lines, [f"component {i}" for i in range(n_components)])
```

## Wavelet Decomposition

```{code-cell} ipython3
:tags: [hide-cell]

from miv.signal.spike import SpikeSorting, WaveletDecomposition
from sklearn.clusterr import MeanShift
```

```{raw-cell}
spike_sorting = SpikeSorting(
    feature_extractor=WaveletDecomposition(),
    clsutering_method=sklearn.cluster.MeanShift()
)
label, index = spike_sorting(cutouts, return_index=True)
```
