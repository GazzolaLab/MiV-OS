__doc__ = """
Connectivity module
"""
__all__ = ["DirectedConnectivity"]  # , 'UndirectedConnectivity']

from typing import Any, List, Optional, Union

import csv
import functools
import gc
import glob
import itertools
import logging
import multiprocessing as mp
import os
import pathlib
import pickle as pkl
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyinform
import pyinform.transferentropy as pyte
import quantities as pq
import scipy.stats as spst
from tqdm import tqdm

from miv.core.datatype import Signal, Spikestamps
from miv.core.operator import OperatorMixin
from miv.core.policy import InternallyMultiprocessing
from miv.core.wrapper import wrap_cacher
from miv.mea import mea_map


# self MPI-able
@dataclass
class DirectedConnectivity(OperatorMixin):
    """
    Directional connectivity analysis operator

    Parameters
    ----------
    bin_size : pq.Quantity
        Bin size for spike train
    mea: Optional[Union[str, np.ndarray]]
        2D array map of MEA channel layout.
    tag : str, optional
        Tag for this operator, by default "directional connectivity analysis"
    progress_bar : bool, optional
        Show progress bar, by default True
    skip_surrogate : bool, optional
        Skip surrogate analysis, by default False
    surrogate_N : int, optional
        Number of surrogate, by default 30
    p_threshold : float, optional
        p-value threshold, by default 0.05
    seed : int, optional
        Random seed. If None, use random seed, by default None
    """

    mea: str = None
    channels: Optional[List[int]] = None
    exclude_channels: Optional[List[int]] = None
    bin_size: float = 0.001
    tag: str = "directional connectivity analysis"
    progress_bar: bool = False

    skip_surrogate: bool = False

    # Surrogate parameters
    surrogate_N: int = 30
    p_threshold: float = 0.05
    H_threshold: float = 1e-2
    seed: int = None
    num_proc: int = 1

    def __post_init__(self):
        super().__init__()
        if isinstance(self.mea, str):
            self.mea_map = mea_map[self.mea]
        else:
            self.mea_map = mea_map["64_intanRHD"]

        if self.exclude_channels is None:  # FIXME: Use dataclass default value
            self.exclude_channels = []

    @wrap_cacher(cache_tag="adjacency_matrix")
    def __call__(self, spikestamps: Spikestamps) -> np.ndarray:
        """__call__.

        Parameters
        ----------
        spikestamps : Spikestamps
        """

        binned_spiketrain: Signal = spikestamps.binning(bin_size=self.bin_size)

        # Channel Selection
        if self.channels is None:
            n_nodes = binned_spiketrain.number_of_channels
            channels = tuple(range(n_nodes))
        else:
            n_nodes = len(self.channels)
            channels = tuple(self.channels)
            binned_spiketrain = binned_spiketrain.select(channels)

        # Get adjacency matrix based on transfer entropy
        adj_matrix = np.zeros([n_nodes, n_nodes], dtype=np.bool_)  # source -> target
        connectivity_metric_matrix = np.zeros(
            [n_nodes, n_nodes], dtype=np.float_
        )  # source -> target

        pairs = [
            (i, j)
            for i, j in itertools.product(range(n_nodes), range(n_nodes))
            if i not in self.exclude_channels and j not in self.exclude_channels
        ]
        func = functools.partial(
            self._get_connection_info,
            binned_spiketrain=binned_spiketrain,
            channels=channels,
            mea=self.mea_map,
            skip_surrogate=self.skip_surrogate,
            surrogate_N=self.surrogate_N,
            seed=self.seed,
            H_threshold=self.H_threshold,
        )
        with mp.Pool(self.num_proc) as pool:
            for idx, result in enumerate(
                tqdm(
                    pool.imap(func, pairs),
                    total=len(pairs),
                    disable=not self.progress_bar,
                )
            ):
                i, j = pairs[idx]
                adj_matrix[i, j] = result[0] < self.p_threshold
                connectivity_metric_matrix[i, j] = result[1]

        connection_ratio = adj_matrix.sum() / adj_matrix.ravel().shape[0]

        info = dict(
            adjacency_matrix=adj_matrix,
            connectivity_matrix=connectivity_metric_matrix,
            connection_ratio=connection_ratio,
        )

        return info

    @staticmethod
    def _surrogate_t_test(
        source, target, skip_surrogate=False, surrogate_N=30, seed=None
    ):
        """
        Surrogate t-test
        """
        # Function configuration. TODO: Make this dependency injection
        te_history = 21
        sublength = 64
        stride = 8

        assert (
            source.shape[0] == target.shape[0]
        ), f"source.shape={source.shape}, target.shape={target.shape}"
        assert (
            source.shape[0] - sublength > 0
        ), f"source.shape[0]={source.shape[0]}, sublength={sublength}"

        rng = np.random.default_rng(seed)  # TODO take rng instead

        func = pyte.transfer_entropy
        te = func(source, target, te_history)
        normalizer = pyinform.entropyrate.entropy_rate(
            target, k=te_history, local=False
        )
        if np.isclose(normalizer, 0.0):
            return 0.0, [0.0]
        te = te / normalizer

        surrogate_tes = []
        if skip_surrogate:
            return te, surrogate_tes
        for _ in range(surrogate_N):
            surrogate_source = source.copy()
            rng.shuffle(surrogate_source)
            for start_index in np.arange(0, source.shape[0] - sublength, stride):
                end_index = start_index + sublength
                surr_te = func(
                    surrogate_source[start_index:end_index],
                    target[start_index:end_index],
                    te_history,
                )
                surrogate_tes.append(surr_te / normalizer)

        return te, surrogate_tes

    @staticmethod
    def _get_connection_info(
        pair,
        binned_spiketrain,
        channels,
        mea,
        skip_surrogate,
        surrogate_N,
        seed,
        H_threshold,
    ):
        """
        Get connection information
        """
        sid, tid = pair
        if sid == tid:
            return 1, 0
        if channels[sid] not in mea or channels[tid] not in mea:
            return 1, 0
        source = binned_spiketrain[sid]
        target = binned_spiketrain[tid]
        te, surrogate_te_list = DirectedConnectivity._surrogate_t_test(
            source,
            target,
            skip_surrogate=skip_surrogate,
            surrogate_N=surrogate_N,
            seed=seed,
        )
        if te < H_threshold:
            return 1, 0
        if skip_surrogate:
            return 1, te
        t_value, p_value = spst.ttest_1samp(surrogate_te_list, te, nan_policy="omit")
        return p_value, te

    def plot_adjacency_matrix(self, result, save_path=None, show=False):
        connectivity_metric_matrix = result["connectivity_matrix"]

        fig, ax = plt.subplots(figsize=(12, 12))
        im = ax.imshow(connectivity_metric_matrix, cmap="gray_r", vmin=0)
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
        ax.set_title("Transfer Entropy")
        plt.colorbar(im, ax=ax)

        if save_path is not None:
            plt.savefig(os.path.join(save_path, "te.png"))

        if show:
            plt.show()

        plt.close(fig)

    def plot_transfer_entropy_histogram(self, result, save_path=None, show=False):
        connectivity_metric_matrix = result["connectivity_matrix"]

        # Export values in csv
        if save_path is not None:
            with open(os.path.join(save_path, "te_values.csv"), mode="w") as f:
                writer = csv.writer(f)
                writer.writerow(["source", "target", "metrics"])

                for i, row in enumerate(connectivity_metric_matrix):
                    for j, value in enumerate(row):
                        writer.writerow([i, j, value])

        # Plot values in heatmap
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.hist(connectivity_metric_matrix, bins=30)
        ax.set_xlabel("transfer entropy")
        ax.set_ylabel("count")
        ax.set_title("Transfer Entropy Histogram")

        if save_path is not None:
            plt.savefig(os.path.join(save_path, "te_histogram.png"))

        if show:
            plt.show()

        plt.close(fig)

    def plot_nodewise_connectivity(
        self,
        result: Any,
        save_path: Union[str, pathlib.Path] = None,
        show: bool = False,
    ):
        """
        Plot nodewise connectivity

        Parameters
        ----------
        result : Any
            Result of this operation.
        save_path : str, optional
            Save path, by default None
        show : bool, optional
            Show plot, by default False

        """
        return
        if show:
            logging.warning(
                "show is not supported for node-wise connectivity plot. Plots will be saved only, if save_path is specified."
            )
            show = False
        if save_path is None:
            return
        adj_matrix = result["adjacency_matrix"]
        connectivity_metric_matrix = result["connectivity_matrix"]

        n_nodes = adj_matrix.shape[0]
        channels = self.channels if self.channels is not None else list(range(n_nodes))

        # Save connectivity plot
        for idx, source in enumerate(channels):
            if source not in self.mea_map:
                continue
            self._plot_directionality(
                adj_matrix[idx],
                source,
                os.path.join(save_path, f"p_graph_source_{source}.png"),
            )
            self._plot_directionality(
                connectivity_metric_matrix[idx],
                source,
                os.path.join(save_path, f"te_graph_source_{source}.png"),
                boolean_connectivity=False,
            )

        for idx, target in enumerate(channels):
            if target not in self.mea_map:
                continue
            self._plot_directionality(
                adj_matrix[:, idx],
                target,
                os.path.join(save_path, f"p_graph_target_{target}.png"),
                reverse=True,
            )
            self._plot_directionality(
                connectivity_metric_matrix[:, idx],
                target,
                os.path.join(save_path, f"te_graph_target_{target}.png"),
                reverse=True,
                boolean_connectivity=False,
            )

    def _plot_directionality(
        self,
        connectivity,
        self_index,
        savename,
        reverse=False,
        boolean_connectivity=True,
    ):
        """
        Plot directionality
        """
        G_1 = nx.DiGraph()

        # position
        pos = {}
        for i in range(self.mea_map.shape[0]):
            for j in range(self.mea_map.shape[1]):
                center = self.mea_map[i, j]
                if center < 0:
                    continue
                G_1.add_node(center)
                pos[center] = (j, -i)

        if boolean_connectivity:
            for idx, conn in enumerate(connectivity):
                if idx == self_index:
                    continue  # No self-connection
                if idx not in self.mea_map:
                    continue  # No connection if node does not exist
                if not conn:
                    continue  # No connection

                if not reverse:
                    G_1.add_edge(self_index, idx)
                else:
                    G_1.add_edge(idx, self_index)
        else:
            for idx, conn in enumerate(connectivity):
                if idx == self_index:
                    continue  # No self-connection
                if idx not in self.mea_map:
                    continue  # No connection if node does not exist

                if not reverse:
                    G_1.add_edge(self_index, idx, weight=conn)
                else:
                    G_1.add_edge(idx, self_index, weight=conn)

        # Numbering
        # edge_weights = [G_1[u][v]['weight'] if 'weight' in G_1[u][v] else 0.0 for u, v in G_1.edges()]
        plt.figure()
        widths = nx.get_edge_attributes(G_1, "weight")
        nx.draw(G_1, pos)
        if boolean_connectivity:
            nx.draw_networkx_edges(
                G_1,
                pos,
                edgelist=widths.keys(),
                ax=plt.gca(),
            )
        else:
            nx.draw_networkx_edges(
                G_1,
                pos,
                edgelist=widths.keys(),
                edge_color=list(widths.values()),
                edge_cmap=plt.cm.gray_r,
                edge_vmin=0.0,
                edge_vmax=1.0,
                ax=plt.gca(),
            )
        nx.draw_networkx_labels(
            G_1,
            pos,
            labels=dict(zip(G_1.nodes(), G_1.nodes())),
            font_color="white",
            ax=plt.gca(),
        )
        # for node, (x_coord, y_coord) in pos.items():
        #    plt.text(x_coord - 0.1, y_coord - 0.1, str(node))

        plt.savefig(savename)
        plt.close()


# TODO
# @dataclass
# class UndirectedConnectivity(OperatorMixin):
#    pass
