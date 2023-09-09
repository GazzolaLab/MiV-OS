__doc__ = """

Typical spike-sorting procedure can be described in three steps: (1) spike detection, (2) feature decomposition, and (3) clustering.
We provide separate module to perform spike-detection; see :ref:`here <api/detection:Spike Detection>`.

We provide `SpikeSorting` module that composes *feature-decomposition* method and *unsupervised-clustering* method.
A commonly used feature-decomposition method includes PCA or wavelet decomposition.
For clustering method, one implemented few commonly appearing methods from the literatures (listed below).
Additionally, one can use out-of-the-box clustering modules from `sklearn`.

.. note:: Depending on the method of clustering, there might be an additional step to find optimum number of cluster.

.. currentmodule:: miv.signal.spike

.. autoclass:: miv.signal.spike.SpikeSorting
   :members:

Available Feature Extractor
###########################

.. autosummary::
   :toctree: _toctree/SpikeSortingAPI

   SpikeFeatureExtractionProtocol
   WaveletDecomposition
   PCADecomposition
   PCAClustering

Unsupervised Clustering
#######################

.. autosummary::
   :toctree: _toctree/SpikeSortingAPI

   UnsupervisedFeatureClusteringProtocol
   SuperParamagneticClustering

Other external tools
--------------------

Following external modules can also be used for the spike sorting.

Sklearn Clustering
~~~~~~~~~~~~~~~~~~
.. autosummary::

   sklearn.cluster.MeanShift
   sklearn.cluster.KMeans

"""
__all__ = [
    "SpikeSorting",
    "PCADecomposition",
    "PCAClustering",
    "WaveletDecomposition",
    "SuperParamagneticClustering",
]

from typing import Any, Optional, Union

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import neo
import numpy as np
import pywt
import quantities as pq
import scipy
import scipy.special
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from miv.core.operator import Operator, OperatorMixin
from miv.signal.spike.protocol import (
    SpikeFeatureExtractionProtocol,
    UnsupervisedFeatureClusteringProtocol,
)
from miv.typing import SignalType, SpikestampsType, TimestampsType


class SpikeSorting:
    """
    Spike sorting module.
    User can specify the method for feature extraction (e.g. WaveletDecomposition, PCADecomposition, etc)
    and the method for clustering (e.g. MeanShift, KMeans, etc).


    Examples
    --------
    >>> spike_sorting = SpikeSorting(
    ...    feature_extractor=PCADecomposition(),
    ...    clustering_method=sklearn.cluster.MeanShift()
    ... )


    Parameters
    ----------
    feature_extractor : SpikeFeatureExtractionProtocol
    clustering_method : UnsupervisedFeatureClusteringProtocol

    """

    def __init__(
        self,
        feature_extractor: SpikeFeatureExtractionProtocol,
        clustering_method: UnsupervisedFeatureClusteringProtocol,
    ):
        self.featrue_extractor = feature_extractor
        self.clustering_method = clustering_method

    def __call__(self, cutouts: np.ndarray, n_group: int = 3):
        assert n_group >= 2, "n_group must be larger than 1"


# UnsupervisedFeatureClusteringProtocol
class SuperParamagneticClustering:  # pragma : no cover
    """Super-Paramagnetic Clustering (SPC)

    The implementation is heavily inspired from [1]_ and [2]_.


    .. [1] Quiroga RQ, Nadasdy Z, Ben-Shaul Y. Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. Neural Comput. 2004 Aug;16(8):1661-87. doi: 10.1162/089976604774201631. PMID: 15228749.
    .. [2] Fernando J. Chaure, Hernan G. Rey, and Rodrigo Quian Quiroga. A novel and fully automatic spike-sorting implementation with variable number of features. Journal of Neurophysiology 2018 120:4, 1859-1871. https://doi.org/10.1152/jn.00339.2018

    """

    def __init__(self):
        raise NotImplementedError


class PCADecomposition:
    """PCA Decomposition

    Other studies that use PCA decomposition: [1]_, [2]_

    .. [1] G. Hilgen, M. Sorbaro, S. Pirmoradian, J.-O. Muthmann, I. Kepiro, S. Ullo, C. Juarez Ramirez, A. Puente Encinas, A. Maccione, L. Berdondini, V. Murino, D. Sona, F. Cella Zanacchi, E. Sernagor, M.H. Hennig (2016). Unsupervised spike sorting for large scale, high density multielectrode arrays. Cell Reports 18, 2521–2532. bioRxiv: http://dx.doi.org/10.1101/048645.
    .. [2] Yger P, Spampinato GL, Esposito E, Lefebvre B, Deny S, Gardella C, Stimberg M, Jetter F, Zeck G, Picaud S, Duebel J, Marre O. A spike sorting toolbox for up to thousands of electrodes validated with ground truth recordings in vitro and in vivo. Elife. 2018 Mar 20;7:e34518. doi: 10.7554/eLife.34518. PMID: 29557782; PMCID: PMC5897014.

    """

    def __init__(self):
        pass

    def project(self, n_components, cutouts, n_features=2):
        scaler = StandardScaler()
        scaled_cutouts = scaler.fit_transform(cutouts)

        pca = PCA()
        pca.fit(scaled_cutouts)
        # print(pca.explained_variance_ratio_)

        pca.n_components = n_features
        transformed = pca.fit_transform(scaled_cutouts)

        # Clustering
        gmm = GaussianMixture(n_components=n_components, n_init=10)
        labels = gmm.fit_predict(transformed)
        return labels, transformed


@dataclass
class PCAClustering(OperatorMixin):
    """
    PCA Clustering Operator
    """

    n_clusters: int = 3
    n_pca_components: int = 2
    tag: str = "pca clustering"

    plot_n_spikes: int = 100

    def __post_init__(self):
        super().__init__()
        self.decomposition = PCADecomposition()

    def __call__(self, cutouts):
        labeled_cutout = {}
        features = {}
        labels_each_channel = {}
        for ch, cutout in cutouts.items():
            if cutout.shape[1] < self.n_clusters:
                continue
            # FIXME: refactor below
            labels, transformed = self.decomposition.project(
                self.n_clusters, cutout.data.T, self.n_pca_components
            )

            cutout_for_each_labels = []
            for i in range(self.n_clusters):
                idx = labels == i
                cutout_for_each_labels.append(cutout.data[:, idx])

            labeled_cutout[ch] = cutout_for_each_labels
            features[ch] = transformed
            labels_each_channel[ch] = labels

        return dict(
            labeled_cutout=labeled_cutout, features=features, labels=labels_each_channel
        )

    def plot_pca_clustered_spikes_all_channels(
        self, output, show=False, save_path=None
    ):
        labeled_cutout = output["labeled_cutout"]
        features = output["features"]
        labels = output["labels"]

        for ch, cutout in labeled_cutout.items():
            if len(cutout) < self.n_clusters:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            for i in range(self.n_clusters):
                idx = labels[ch] == i
                axes[0].plot(
                    features[ch][idx, 0], features[ch][idx, 1], ".", label=f"group {i}"
                )

            for label in range(self.n_clusters):
                color = plt.rcParams["axes.prop_cycle"].by_key()["color"][label]
                for i in range(min(self.plot_n_spikes, cutout[label].shape[1])):
                    axes[1].plot(
                        # time,
                        cutout[label][:, i],
                        alpha=0.3,
                        linewidth=1,
                        color=color,
                    )
            axes[0].set_title("Cluster assignments by a GMM")
            axes[0].set_xlabel("Principal Component 1")
            axes[0].set_ylabel("Principal Component 2")
            axes[0].legend()
            axes[0].axis("tight")

            axes[1].set_xlabel("Time (ms)")
            axes[1].set_ylabel("Voltage (microV)")
            axes[1].set_title("Spike Cutouts")

            # custom legend
            custom_lines = [
                plt.Line2D(
                    [0],
                    [0],
                    color=plt.rcParams["axes.prop_cycle"].by_key()["color"][i],
                    lw=4,
                )
                for i in range(self.n_clusters)
            ]
            axes[1].legend(
                custom_lines, [f"component {i}" for i in range(self.n_clusters)]
            )

            if save_path is not None:
                plt.savefig(os.path.join(save_path, f"pca_channel{ch}.png"))
            plt.close(fig)
            plt.close("all")


class WaveletDecomposition:  # TODO
    """
    Wavelet Decomposition for spike sorting.
    The implementation is heavily inspired from [1]_ and [2]_;
    their MatLab implementation (wave_clus) can be found `here <https://github.com/csn-le/wave_clus>`_.

    The default setting uses four-level multiresolution decomposition with Haar wavelets.
    To learn about possible choice of wavelet, check `PyWavelets module <https://pywavelets.readthedocs.io/en/latest/#>`_.

    Other studies that use wavelet decomposition: [3]_

    .. [1] Letelier JC, Weber PP. Spike sorting based on discrete wavelet transform coefficients. J Neurosci Methods. 2000 Sep 15;101(2):93-106. doi: 10.1016/s0165-0270(00)00250-8. PMID: 10996370.
    .. [2] Quiroga RQ, Nadasdy Z, Ben-Shaul Y. Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. Neural Comput. 2004 Aug;16(8):1661-87. doi: 10.1162/089976604774201631. PMID: 15228749.
    .. [3] Nenadic, Z., and Burdick, J. W. (2005). Spike detection using the continuous wavelet transform. IEEE Trans. BioMed. Eng. 52, 74–87. doi: 10.1109/TBME.2004.839800

    """

    def __init__(self):
        pass

    def project(self, n_features):
        ## Wavelet Decomposition
        number_of_spikes = 400
        data_length = 100
        cutouts = np.empty([number_of_spikes, data_length])
        # spikes_l = cutouts[0]
        coeffs = pywt.wavedec(cutouts, "haar", level=4)
        features = np.concatenate(coeffs, axis=1)

        def test_ks(x):
            # Calculates CDF

            # xCDF = []
            yCDF = []
            x = x[~np.isnan(x)]
            n = x.shape[0]
            x.sort()

            # Get cumulative sums
            yCDF = (np.arange(n) + 1) / n

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
            theocdf = 0.5 * scipy.special.erfc(-(zScore - mu) / (np.sqrt(2) * sigma))

            # Compute the Maximum distance: max|S(x) - theocdf(x)|.

            delta1 = (
                y_expcdf[:-1] - theocdf
            )  # Vertical difference at jumps approaching from the LEFT.
            delta2 = (
                y_expcdf[1:] - theocdf
            )  # Vertical difference at jumps approaching from the RIGHT.
            deltacdf = np.abs(np.concatenate([delta1, delta2]))

            KSmax = deltacdf.max()
            return KSmax

        ks = []
        for idx, feature in enumerate(np.moveaxis(features, 1, 0)):
            std_feature = np.std(feature)
            mean_feature = np.mean(feature)
            thr_dist = std_feature * 3
            thr_dist_min = mean_feature - thr_dist
            thr_dist_max = mean_feature + thr_dist
            aux = feature[
                np.logical_and(feature > thr_dist_min, feature < thr_dist_max)
            ]

            if aux.shape[0] > 10:
                ks.append(test_ks(aux))
            else:
                ks.append(0)

        max_inputs = 0.75
        min_inputs = 10

        # if all:
        # max_inputs = features.shape[1]
        if max_inputs < 1:
            max_inputs = np.ceil(max_inputs * features.shape[1]).astype(int)

        ind = np.argsort(ks)
        A = np.array(ks)[ind]
        A = A[A.shape[0] - max_inputs :]  # Cutoff coeffs

        ncoeff = A.shape[0]
        maxA = A.max()
        nd = 10
        d = (A[nd - 1 :] - A[: -nd + 1]) / maxA * ncoeff / nd
        all_above1 = d[np.nonzero(d >= 1)]
        if all_above1.shape[0] >= 2:
            # temp_bla = smooth(diff(all_above1),3)
            aux2 = np.diff(all_above1)
            temp_bla = np.convolve(aux, np.ones(3) / 3)
            temp_bla = temp_bla[1:-1]
            temp_bla[0] = aux2[0]
            temp_bla[-1] = aux2[-1]
            # ask to be above 1 for 3 consecutive coefficients
            thr_knee_diff = all_above1[np.nonzero(temp_bla[1:] == 1)[:1]] + nd / 2
            inputs = max_inputs - thr_knee_diff + 1
        else:
            inputs = min_inputs

        plot_feature_stats = True
        if plot_feature_stats:
            fig = plt.figure()
            plt.stairs(np.sort(ks))
            plt.plot(
                [len(ks) - inputs + 1, len(ks) - inputs + 1],
                fig.axes[0].get_ylim(),
                "r",
            )
            plt.plot(
                [len(ks) - max_inputs, len(ks) - max_inputs],
                fig.axes[0].get_ylim(),
                "--k",
            )
            plt.ylabel("ks_stat")
            plt.xlabel("# features")
            plt.title(
                f"number of spikes = {number_of_spikes}, inputs_selected = {inputs}"
            )

        if inputs > max_inputs:
            inputs = max_inputs
        elif inputs.shape[0] == 0 or inputs < min_inputs:
            inputs = min_inputs

        coeff = ind[-inputs:]
        # CRATES INPUT MATRIX FOR SPC
        input_for_spc = np.zeros((number_of_spikes, inputs))

        for i in range(number_of_spikes):
            for j in range(inputs):
                input_for_spc[i, j] = features[i, coeff[j]]
