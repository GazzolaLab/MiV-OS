---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3.10.4 64-bit
  language: python
  name: python3
---

<h3> This notebook shows that the bins_per_second parameter for auto_channel_mask should negligible affect the result of channel masking.
</h3>

```{code-cell} ipython3
from miv.io import*
from miv.signal.filter import ButterBandpass
from miv.signal.spike import ThresholdCutoff

folder_path: str = './2022-03-10_16-19-09'

band_filter = ButterBandpass(300, 3000)
threshold_detector = ThresholdCutoff()
```

```{code-cell} ipython3
data_man = DataManager(folder_path)
with data_man[1].load() as (sig, times, samp):
    filteredSig = band_filter(sig, samp)
    exp2Spikes = threshold_detector(filteredSig, times, samp)
```

```{code-cell} ipython3
from viziphant.rasterplot import*
rasterplot_rates(exp2Spikes, histscale=0.5, markerargs={'marker':'.','markersize': 0.5})
```

The rasterplot and histogram above shows the spikes of the unmasked band-filtered signals. 
___
Now, we call the auto_channel_mask method with various bins_per_second values:

```{code-cell} ipython3
data_man1 = DataManager(folder_path)
data_man10 = DataManager(folder_path)
data_man100 = DataManager(folder_path)
data_man1000 = DataManager(folder_path)
data_man10000 = DataManager(folder_path)
data_man100000 = DataManager(folder_path)
```

```{code-cell} ipython3
data_man1.auto_channel_mask_v4(band_filter, threshold_detector, bins_per_second=1)
data_man10.auto_channel_mask_v4(band_filter, threshold_detector, bins_per_second=10)
data_man100.auto_channel_mask_v4(band_filter, threshold_detector, bins_per_second=100)
data_man1000.auto_channel_mask_v4(band_filter, threshold_detector, bins_per_second=1000)
data_man10000.auto_channel_mask_v4(band_filter, threshold_detector, bins_per_second=10000)
data_man100000.auto_channel_mask_v4(band_filter, threshold_detector, bins_per_second=100000)
```

The greater the number of bins per second, the more computation and memory is needed. This should be O(n<sup>3</sup>) since matrix multiplication is involved.

```{code-cell} ipython3
data_men = [data_man1, data_man10, data_man100, data_man1000, data_man10000, data_man100000]
masked_spikes = []
for man in data_men:
    with man[2].load() as (sig, times, samp):
        masked_spikes.append(threshold_detector(sig, times, samp))
```

```{code-cell} ipython3
from viziphant.rasterplot import*
rasterplot_rates(masked_spikes, histscale=0.5, markerargs={'marker':'.','markersize': 0.5})
```

We can see from the above rasterplot that the spikes look identical. The histograms completely overlap.
