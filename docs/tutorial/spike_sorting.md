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

# Build-Your-Own Operator: Spike Sorting

Here we will build a simple spike sorting operator using the `miv` library. We will use the `miv` library to load the data, filter the signal, and extract the spikes. We will then use `scikit-learn` to cluster the spikes.

To learn more about the spike-sorting techniques, refer to the following:

- Spike sorting based on discrete wavelet transform coefficients (Letelier 2000)
- Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering (Quiroga 2004)
- A novel and fully automatic spike-sorting implementation with variable number of features (Chaure 2018)

We start with importing the dataset and the necessary libraries.

```{code-cell} ipython3
:tags: [hide-cell]

import numpy as np
import quantities as pq
import matplotlib.pyplot as plt

from miv.core import Pipeline
from miv.io.openephys import DataManager
from miv.signal import ButterBandpass, ThresholdCutoff, ExtractWaveforms

from miv.datasets.openephys_sample import load_data

# Prepare data
dataset = load_data(progbar_disable=True)
data = dataset[0]
```

## Obtain Spike Cutouts

Please refer to the previous tutorial on spike detection. Here, we'll further process using the obtained cutouts data.

```{code-cell} ipython3
bandpass_filter = ButterBandpass(lowcut=400, highcut=1500, order=4)
spike_detection = ThresholdCutoff()
extract_waveforms = ExtractWaveforms(channels=[11, 26, 37, 50], plot_n_spikes=None)

data >> bandpass_filter >> spike_detection
bandpass_filter >> extract_waveforms
spike_detection >> extract_waveforms

fig, ax = plt.subplots()
data.visualize(ax)
plt.show()
```

## Build Operator

As an example, let's build a simple operator that will print out the shape of the cutouts.
The easiest way to build an operator is to make `dataclass` and inherit the `OperatorMixin` class.
The `OperatorMixin` allow us to use the `dataclass` object as a part of a `Pipeline`.

The mixin `OperatorMixin` provides most of the necessary variables and methods to build an operator, except for the `tag`. The `tag` is used to identify the operator, and use to name the output files and directory. To initialize the operator, we will use the `__post_init__` method.

```{code-cell} ipython3
from dataclasses import dataclass
from miv.core import OperatorMixin

@dataclass
class CutoutShape(OperatorMixin):
    tag = "cutout shape"

    def __post_init__(self):
        super().__init__()

    def __call__(self, cutouts):
        shapes = []
        for channel, cutout in cutouts.items():
            shapes.append(cutout.shape)
        return shapes

    def after_run_print(self, output):
        print(output)
        return output

cutout_shape = CutoutShape()
extract_waveforms >> cutout_shape

fig, ax = plt.subplots()
data.visualize(ax)
plt.show()
```

The operator `CutoutShape` will print out the shape of the cutouts.
Notice, the function `__call__` is the operation that transforms the input data `cutouts` to the output data. The return type of the previous operator `extract_waveforms` is a dictionary of `cutouts` with the channel number as the key. The `CutoutShape` operator will return a list of tuples of the shape of the cutouts.
The print is taken care of by the `after_run_print` method.

```{note}
Any function with the name starting with `after_run` will be executed after the `__call__`.
Similarly, any function with the name starting with `plot` will be executed during the plotting.
```

We can then create a `Pipeline` object and run it to execute the processing steps.

```{code-cell} ipython3
pipeline = Pipeline(cutout_shape)
pipeline.run()
```

## PCA and Gaussian Mixture Clustering

Now, we can use the above template to construct a more complex operator that will perform PCA and Gaussian Mixture Clustering on the cutouts.

```{code-cell} ipython3
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

@dataclass
class PCAClustering(OperatorMixin):
    n_clusters: int = 3
    n_pca_components: int = 2
    tag = "pca clustering"

    plot_n_spikes: int = 100

    def __post_init__(self):
        super().__init__()

    def __call__(self, cutouts):
        labeled_cutout = {}
        features = {}
        labels_each_channel = {}
        for ch, cutout in cutouts.items():
            # Standardize
            scaler = StandardScaler()
            scaled_cutouts = scaler.fit_transform(cutout.data.T)

            # PCA
            pca = PCA()
            pca.fit(scaled_cutouts)
            pca.n_components = self.n_pca_components
            transformed = pca.fit_transform(scaled_cutouts)

            # GMM Clustering
            gmm = GaussianMixture(n_components=self.n_clusters, n_init=10)
            labels = gmm.fit_predict(transformed)

            cutout_for_each_labels = []
            for i in range(self.n_clusters):
                idx = labels == i
                cutout_for_each_labels.append(cutout.data[:,idx])

            labeled_cutout[ch] = cutout_for_each_labels
            features[ch] = transformed
            labels_each_channel[ch] = labels

        return dict(labeled_cutout=labeled_cutout, features=features, labels=labels_each_channel)

    def plot_pca_clustered_spikes_all_channels(self, outputs, inputs, show=False, save_path=None):
        labeled_cutout = outputs["labeled_cutout"]
        features = outputs["features"]
        labels = outputs["labels"]

        for ch, cutout in labeled_cutout.items():
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            for i in range(self.n_clusters):
                idx = labels[ch] == i
                axes[0].plot(features[ch][idx, 0], features[ch][idx, 1], ".", label=f"group {i}")


            #time = signal.timestamps * pq.s
            #time = time.rescale(pq.ms).magnitude
            for label in range(self.n_clusters):
                color = plt.rcParams["axes.prop_cycle"].by_key()["color"][label]
                for i in range(min(self.plot_n_spikes, cutout[label].shape[1])):
                    axes[1].plot(
                        #time,
                        cutout[label][:, i],
                        alpha=0.3,
                        linewidth=1,
                        color=color
                    )
        axes[0].set_title("Cluster assignments by a GMM")
        axes[0].set_xlabel("Principal Component 1")
        axes[0].set_ylabel("Principal Component 2")
        axes[0].legend()
        axes[0].axis("tight")

        axes[1].set_xlabel("Time (ms)")
        axes[1].set_ylabel("Voltage (mV)")
        axes[1].set_title(f"Spike Cutouts")

        # custom legend
        custom_lines = [plt.Line2D([0], [0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"][i], lw=4,) \
                            for i in range(self.n_clusters)]
        axes[1].legend(custom_lines, [f"component {i}" for i in range(self.n_clusters)])

cluster = PCAClustering()
extract_waveforms >> cluster

fig, ax = plt.subplots()
data.visualize(ax, seed=150)
plt.show()
```

## Run Pipeline

We can then create a `Pipeline` object and run it to execute the processing steps.
All the result files will be saved in the `results` directory.

```{code-cell} ipython3
pipeline = Pipeline(cluster)
print(pipeline.summarize())
pipeline.run("results")
```
