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

## Wavelet Decomposition

```{raw-cell}
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

```{code-cell} ipython3
spikes_l = cutouts[0]
coeffs = pywt.wavedec(cutouts, 'haar', level=4)
features = np.concatenate(coeffs, axis=1)
```

```{code-cell} ipython3
def test_ks(x):
    # Calculates CDF

    xCDF, yCDF = [], []
    x = x[~np.isnan(x)]
    n = x.shape[0]
    x.sort()

    # Get cumulative sums
    yCDF = (np.arange(n)+1) / n

    # Remove duplicates; only need final one with total count
    notdup = np.concatenate([np.diff(x), [1]]) > 0
    x_expcdf = x[notdup]
    y_expcdf = np.concatenate([[0], yCDF[notdup]])

    # The theoretical CDF (theocdf) is assumed to ben ormal
    # with unknown mean and sigma
    zScore = (x_expcdf - x.mean()) / x.std()
    # theocdf = normcdf(zScore, 0, 1)

    mu = 0
    sigma = 1
    theocdf = 0.5 * scipy.special.erfc(-(zScore-mu)/(np.sqrt(2)*sigma))

    # Compute the Maximum distance: max|S(x) - theocdf(x)|.

    delta1 =  y_expcdf[ :-1] - theocdf  # Vertical difference at jumps approaching from the LEFT.
    delta2 =  y_expcdf[1:  ] - theocdf  # Vertical difference at jumps approaching from the RIGHT.
    deltacdf = np.abs(np.concatenate([delta1, delta2]))

    KSmax = deltacdf.max()
    return KSmax
```

```{code-cell} ipython3
ks = []
for idx, feature in enumerate(np.moveaxis(features, 1, 0)):
    std_feature = np.std(feature)
    mean_feature = np.mean(feature)
    thr_dist = std_feature * 3;
    thr_dist_min = mean_feature - thr_dist;
    thr_dist_max = mean_feature + thr_dist;
    aux = feature[np.logical_and(feature>thr_dist_min, feature<thr_dist_max)];

    if aux.shape[0] > 10:
        ks.append(test_ks(aux))
    else:
        ks.append(0)
```

```{code-cell} ipython3
max_inputs = 0.75
min_inputs = 10

#if all:
#max_inputs = features.shape[1]
if max_inputs < 1:
    max_inputs = np.ceil(max_inputs * features.shape[1]).astype(int)

ind = np.argsort(ks)
A = np.array(ks)[ind]
A = A[A.shape[0] - max_inputs:]  # Cutoff coeffs

ncoeff = A.shape[0]
maxA = A.max()
nd = 10
d = (A[nd-1:] - A[:-nd+1]) / maxA * ncoeff / nd
all_above1 = d[np.nonzero(d>=1)]
if all_above1.shape[0] >= 2:
    # temp_bla = smooth(diff(all_above1),3)
    aux2 = np.diff(all_above1)
    temp_bla = np.convolve(aux, np.ones(3)/3)
    temp_bla = temp_bla[1:-1]
    temp_bla[0] = aux2[0]
    temp_bla[-1] = aux2[-1]
    # ask to be above 1 for 3 consecutive coefficients
    thr_knee_diff = all_above1[np.nonzero(temp_bla[1:] == 1)[:1]] + nd/2
    inputs = max_inputs - thr_knee_diff + 1
else:
    inputs = min_inputs
```

```{code-cell} ipython3
plot_feature_stats = True
if plot_feature_stats:
    fig = plt.figure()
    plt.stairs(np.sort(ks))
    plt.plot([len(ks)-inputs+1, len(ks)-inputs+1], fig.axes[0].get_ylim(), 'r')
    plt.plot([len(ks)-max_inputs, len(ks)-max_inputs], fig.axes[0].get_ylim(), '--k')
    plt.ylabel('ks_stat')
    plt.xlabel('# features')
    plt.title(f"number of spikes = {number_of_spikes}, inputs_selected = {inputs}")


```

```{code-cell} ipython3
if inputs > max_inputs:
    inputs = max_inputs
elif inputs.size == 0 or inputs < min_inputs:
    inputs = min_inputs
```

```{code-cell} ipython3
coeff = ind[-inputs:];
# CRATES INPUT MATRIX FOR SPC
input_for_spc = np.zeros((number_of_spikes, inputs))

for i in range(number_of_spikes):
    for j in range(inputs):
        input_for_spc[i,j] = features[i, coeff[j]]
```
