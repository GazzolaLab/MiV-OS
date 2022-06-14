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

```{code-cell} ipython3
import os, sys
import miv
import scipy.signal as ss

from miv.io import DataManager
from miv.signal.filter import ButterBandpass, MedianFilter, FilterCollection
from miv.signal.spike import ThresholdCutoff, SpikeSorting, PCADecomposition
from miv.statistics import signal_to_noise, spikestamps_statistics
from miv.visualization import plot_frequency_domain, extract_waveforms, plot_waveforms

import elephant.statistics
from viziphant.rasterplot import rasterplot_rates

from tqdm import tqdm

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
```

```{code-cell} ipython3
REANALYZE = True
PLOT_SPIKE_CUTOUT_PCA = True
PLOT_RATE_DATA = True

# Configuration
#data_paths = ["2022-05-15_14-51-36","2022-05-17_12-52-34","2022-05-19_13-27-29","2022-05-21_13-33-51","2022-05-23_14-23-46","2022-05-25_15-12-57"]
data_paths = ["2022-05-25_15-12-57"]
#experiment_query = ["experiment1_","experiment2_","experiment3_"]
experiment_query = ["experiment_std2_stim"]#,"experiment10","experiment11"]


# Data call
signal_filter = (
    FilterCollection()
        .append(ButterBandpass(600, 2400, order=4))
        .append(MedianFilter(threshold=60, k=20))
)
spike_detection = ThresholdCutoff(cutoff=5)
# Data call
experiment_rates = []
if REANALYZE:
    for q_idx, query in enumerate(experiment_query):
        rates = []
        for path in data_paths:
            data_collection = DataManager(path)
            data = data_collection.query_path_name(query)[0]

            with data.load() as (signal, timestamps, sampling_rate):
                #data.set_channel_mask(range(10,50))
                # Preprocess
                signal = signal_filter(signal, sampling_rate)
                spiketrains = spike_detection(signal, timestamps, sampling_rate)

                # Firing Rate
                stat = spikestamps_statistics(spiketrains)['rates']

                if PLOT_SPIKE_CUTOUT_PCA:
                    # Cutout
                    for ch in tqdm(range(signal.shape[1])):
                        if len(spiketrains[ch]) < 5: continue
                        cutouts = extract_waveforms(
                            signal, spiketrains, channel=ch, sampling_rate=sampling_rate
                        )
                        fig = plt.figure(figsize=(12,6))
                        plot_waveforms(cutouts, sampling_rate, n_spikes=300)
                        data.save_figure(fig, "cutout", f"ch_{ch}.png")
                        plt.cla()
                        plt.clf()
                        plt.close("all")

                        # PCA+GMM (TODO: refactor using MIV-OS)
                        pca = PCADecomposition()
                        n_components = 5
                        labels, transformed = pca.project(n_components, cutouts)
                        """
                        spike_sorting = SpikeSorting(
                            feature_extractor=PCADecomposition(),
                            clustering_method=sklearn.cluster.KMean()
                        )
                        label, index = spike_sorting(cutouts, return_index=True)
                        """

                        fig = plt.figure(figsize=(8, 8))
                        for i in range(n_components):
                            idx = labels == i
                            _ = plt.plot(transformed[idx, 0], transformed[idx, 1], ".")
                            _ = plt.title("Cluster assignments by a GMM")
                            _ = plt.xlabel("Principal Component 1")
                            _ = plt.ylabel("Principal Component 2")
                            _ = plt.legend([0, 1, 2])
                            _ = plt.axis("tight")
                        data.save_figure(fig, "pca_cluster", f"PCA_ch_{ch}.png")
                        plt.cla()
                        plt.clf()
                        plt.close("all")

                        fig = plt.figure(figsize=(8, 8))
                        for i in range(n_components):
                            idx = (labels == i)
                            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
                            plot_waveforms(
                                cutouts[idx, :], sampling_rate, n_spikes=250, color=color,
                            )
                        # custom legend
                        # fmt: off
                        custom_lines = [plt.Line2D([0], [0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0], lw=4,),
                                        plt.Line2D([0], [0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1], lw=4,),
                                        plt.Line2D([0], [0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"][2], lw=4,)]
                        # fmt: on
                        plt.legend(custom_lines, ["0", "1", "2"])
                        data.save_figure(fig, "pca_cutputs", f"ch_{ch}.png")
                        plt.cla()
                        plt.clf()
                        plt.close("all")

            rates.append(stat)
        experiment_rates.append(rates)
    experiment_rates = np.array(experiment_rates)

    #False Spike Channels
    experiment_rates[:,:,12] = 0
    experiment_rates[:,:,41] = 0
    experiment_rates[:,:,42] = 0
    experiment_rates[:,:,43] = 0
    experiment_rates[:,:,45] = 0
    experiment_rates[:,:,47] = 0
    experiment_rates[:,:,36] = 0
    experiment_rates[:,:,53] = 0
    experiment_rates[:,:,57] = 0
    experiment_rates[:,:,55] = 0
    experiment_rates[:,:,61] = 0
    experiment_rates[:,:,62] = 0
    experiment_rates[:,:,15] = 0
    experiment_rates[:,:,58] = 0
    np.savez("rates.npz", rates=experiment_rates)

    ## PCA


    ## Cutout

else:
    experiment_rates = np.load("rates.npz")["rates"]
    print(experiment_rates.shape)
```

```{code-cell} ipython3
width = 0.8
x = np.arange(64)
for q_idx, query in enumerate(experiment_query):
    fig, ax = plt.subplots(figsize=(16,8))
    for p_idx, path in enumerate(data_paths):
        rates = experiment_rates[q_idx][p_idx]
        ax.bar(1+x-width/2+(p_idx*2)*width/len(data_paths), rates, width, label=path, align="edge")
    ax.set_xlabel('Channel')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('Firing Rate Progression')
    lower =1
    upper =64
    length = 64
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{query}_firing_rate_bar_unfiltered.png')
```

```{code-cell} ipython3
# To plot Active Channels
PP = np.array(experiment_rates)
BP = np.array(experiment_rates)
BP[PP<0.2]=0
BP[PP>0.2]=1
plt.plot(BP[0,:],'-o')
plt.plot(BP[1,:],'-o')
plt.plot(BP[2,:],'-o')
plt.xticks([0,1,2,3,4],[9,11,13,15,17])
plt.legend(['Control', 'Standard1','Standard2'])
plt.xlabel("Days After Plating")
plt.ylabel("Number Of Active Channels")
plt.savefig('Active_Channels.png',dpi=600)
```

```{code-cell} ipython3
PLOT_SPIKE_CUTOUT_PCA = True
PLOT_RATE_DATA = True

if PLOT_RATE_DATA:
    # Visualize Rate

    ## Bar chart (unfiltered)
    width = 0.8
    x = np.arange(64)*3
    for q_idx, query in enumerate(experiment_query):
        fig, ax = plt.subplots(figsize=(16,8))
        for p_idx, path in enumerate(data_paths):
            rates = experiment_rates[q_idx][p_idx]
            ax.bar(x-width/2+(p_idx*3)*width/len(data_paths), rates, width, label=path, align="edge")
        ax.set_xlabel('Channel')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title('Firing Rate Progression')
        ax.set_xticks(x, [f"{ch+1}" for ch in range(len(x))])
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'{query}_firing_rate_bar_unfiltered.png')

    ## Violinplot
    for q_idx, query in enumerate(experiment_query):
        fig, ax = plt.subplots()
        raw_data = experiment_rates[q_idx]
        data = [d[d>1] for d in raw_data]
        ax.violinplot(data, showmeans=True,)
        ax.set_xlabel('Date')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title('Firing Rate Progression (violin plot)')
        ax.set_xticks(np.arange(len(data_paths))+1, data_paths)
        fig.tight_layout()
        fig.savefig(f'{query}_violin.png')

    ## 3D Histogram
    mea_map = np.array([[-1  , 10 , 12 , 14 , 31 , 28 , 26 , -1],
                         [18 , 17 , 11 , 13 , 32 , 27 , 38 , 37],
                         [20 , 19 , 9  , 15 , 30 , 39 , 36 , 35],
                         [23 , 22 , 21 , 16 , 29 , 34 , 33 , 56],
                         [-1 , 1  , 2  , 61 , 44 , 53 , 54 , 55],
                         [3  , 4  , 7  , 62 , 43 , 48 , 51 , 52],
                         [5  , 6  , 59 , 64 , 41 , 46 , 49 , 50],
                         [-1 , 58 , 60 , 63 , 42 , 45 , 47 , -1]])
    for q_idx, query in enumerate(experiment_query):
        plot_n_col = len(data_paths)
        fig = plt.figure(figsize=(16,8 * plot_n_col))
        for p_idx, path in enumerate(data_paths):
            firing_map = np.zeros([8,8])
            rates = experiment_rates[q_idx][p_idx]
            for ch_idx, rate in enumerate(rates):
                locx, locy = np.where(mea_map == ch_idx)
                firing_map[locx, locy] = rate
            locx, locy = np.where(mea_map == -1)

            # 2D
            ax1 = fig.add_subplot(plot_n_col,2,1+2*p_idx)
            im = ax1.imshow(firing_map, cmap="Oranges")
            cbar = ax1.figure.colorbar(im, ax=ax1)
            cbar.ax.set_ylabel("Firing Rate", rotation=-90, va="bottom")
            ax1.set_xlabel("column")
            ax1.set_ylabel("row")
            ax1.set_title("Firing Rate Heat Map")
            # 3D
            ax2 = fig.add_subplot(plot_n_col,2,2+2*p_idx, projection='3d')
            _x = np.arange(8)
            _y = np.arange(8)
            _xx, _yy = np.meshgrid(_x, _y)
            x, y = _xx.ravel(), _yy.ravel()
            bottom = 0
            width = depth = 1
            #color
            dz = firing_map[::-1,:].ravel()
            offset = dz+ np.abs(dz.min())
            fracs = offset.astype(float)/offset.max()
            norm = colors.Normalize(fracs.min(), fracs.max())
            colors_scheme = cm.get_cmap("Oranges")(norm(fracs.tolist()))
            #bar3d
            ax2.bar3d(x,y,bottom,width,depth,dz, shade=True, color=colors_scheme)
            ax2.set_zlim(firing_map.min(), firing_map.max())
            ax2.set_xlabel("column")
            ax2.set_ylabel("row")
            ax2.set_title("Firing Rate 3D Histogram")

        fig.tight_layout()
        fig.savefig(f"{query}_firing_rate_over_mea.png")
```

```{code-cell} ipython3
#RasterPlot

a,b,c = rasterplot_rates(spiketrains)
a.set_xlim(0,30)
#plt.show()
plt.savefig('Raster_plot_Std2_test',dpi=600)
```

```{code-cell} ipython3
#Cell_Assembly_Detection

import quantities as pq
import viziphant
from elephant.cell_assembly_detection import cell_assembly_detection

rates = elephant.statistics.BinnedSpikeTrain(spiketrains, bin_size=100*pq.ms)
rates.rescale('ms')
patterns = cell_assembly_detection(rates, max_lag=2)


a= viziphant.patterns.plot_patterns(spiketrains, patterns=patterns[:5],
                                 circle_sizes=(1, 10, 10))

a.set_xlim(0,10)
a.set_ylim(0,20)
bb=np.linspace(0,20,5)
a.set_ylabel('')
a.set_yticks(bb)
#plt.show()
plt.savefig('Raster_plot_Std2_Synchronous_stim',dpi=600)
```

```{code-cell} ipython3
#Cross_Correlation Matrix
from elephant.spike_train_correlation import correlation_coefficient
from viziphant.spike_train_correlation import plot_corrcoef

corrcoef_matrix = correlation_coefficient(rates)

fig, axes = plt.subplots()
plot_corrcoef(corrcoef_matrix, axes=axes)
axes.set_xlabel('Electrode')
axes.set_ylabel('Electrode')
axes.set_title("Correlation coefficient matrix")
plt.show()
```
