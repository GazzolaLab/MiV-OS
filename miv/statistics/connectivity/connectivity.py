__doc__ = """
Connectivity module
"""
__all__ = ["DirectedConnectivity"]  # , 'UndirectedConnectivity']

from typing import Any, Optional, Union

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

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyinform.transferentropy as pyte
import quantities as pq
import scipy.stats as spst
from tqdm import tqdm

from miv.core.datatype import Signal, Spikestamps
from miv.core.operator import OperatorMixin
from miv.core.wrapper import wrap_generator_to_generator
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
    mea_map_key : str
        MEA map key
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

    mea_map_key: str
    bin_size: float = 1.0
    tag: str = "directional connectivity analysis"
    progress_bar: bool = True

    skip_surrogate: bool = False

    # Surrogate parameters
    surrogate_N: int = 30
    p_threshold: float = 0.05
    seed: int = None

    def __post_init__(self):
        super().__init__()
        self.mea = mea_map[self.mea_map_key]

    @wrap_generator_to_generator
    def __call__(self, spikestamps: Spikestamps) -> np.ndarray:
        """__call__.

        Parameters
        ----------
        spikestamps : Spikestamps
        """

        binned_spiketrain: Signal = spikestamps.binning(bin_size=self.bin_size)
        n_nodes = binned_spiketrain.number_of_channels

        # Get adjacency matrix based on transfer entropy
        adj_matrix = np.zeros([n_nodes, n_nodes], dtype=np.bool_)  # source -> target
        connectivity_metric_matrix = np.zeros(
            [n_nodes, n_nodes], dtype=np.float_
        )  # source -> target
        # TODO: Use mp.Pool
        for source in tqdm(range(n_nodes), disable=not self.progress_bar):
            if source not in self.mea:
                continue
            p_values = []
            metric_values = []
            for target in tqdm(
                range(n_nodes), disable=not self.progress_bar, leave=False
            ):
                if source != target and target in self.mea:
                    source_binned_spiketrain = binned_spiketrain[source]
                    target_binned_spiketrain = binned_spiketrain[target]
                    p, metrics = self._get_connection_info(
                        source_binned_spiketrain, target_binned_spiketrain
                    )
                else:
                    p, metrics = 1, 0
                p_values.append(p)
                metric_values.append(metrics)
            adj_matrix[source] = np.array(p_values) < self.p_threshold
            connectivity_metric_matrix[source] = np.array(metric_values)
        connection_ratio = adj_matrix.sum() / adj_matrix.ravel().shape[0]

        info = dict(
            adjacency_matrix=adj_matrix,
            connectivity_matrix=connectivity_metric_matrix,
            connection_ratio=connection_ratio,
        )

        return info

    def _surrogate_t_test(self, source, target, te_history=16, sublength=64, stride=8):
        """
        Surrogate t-test
        """
        assert source.shape[0] == target.shape[0]
        assert source.shape[0] - sublength > 0

        rng = np.random.default_rng(self.seed)  # TODO take rng instead

        tes = []
        for start_index in np.arange(0, source.shape[0] - sublength, stride):
            end_index = start_index + sublength
            te = pyte.transfer_entropy(
                source[start_index:end_index], target[start_index:end_index], te_history
            )
            tes.append(te)

        surrogate_tes = []
        if not self.skip_surrogate:
            for _ in range(self.surrogate_N):
                surrogate_source = source.copy()
                rng.shuffle(surrogate_source)
                for start_index in np.arange(0, source.shape[0] - sublength, stride):
                    end_index = start_index + sublength
                    surr_te = pyte.transfer_entropy(
                        surrogate_source[start_index:end_index],
                        target[start_index:end_index],
                        te_history,
                    )
                    surrogate_tes.append(surr_te)

        return tes, surrogate_tes

    def _get_connection_info(self, source, target):
        """
        Get connection information
        """
        te_list, surrogate_te_list = self._surrogate_t_test(source, target)
        t_value, p_value = spst.ttest_ind(
            te_list, surrogate_te_list, equal_var=False, nan_policy="omit"
        )
        return p_value, np.mean(te_list)

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
        if show:
            logging.warning("show is not supported for this plot.")
        adj_matrix = result["adjacency_matrix"]
        connectivity_metric_matrix = result["connectivity_matrix"]

        n_nodes = adj_matrix.shape[0]

        # Save connectivity plot
        for source in range(n_nodes):
            if source not in self.mea:
                continue
            self._plot_directionality(
                adj_matrix[source],
                source,
                n_nodes,
                os.path.join(save_path, f"p_graph_source_{source}.png"),
            )
            self._plot_directionality(
                connectivity_metric_matrix[source],
                source,
                n_nodes,
                os.path.join(save_path, f"te_graph_source_{source}.png"),
                boolean_connectivity=False,
            )

        for target in range(n_nodes):
            if target not in self.mea:
                continue
            self._plot_directionality(
                adj_matrix[:, target],
                target,
                n_nodes,
                os.path.join(save_path, f"p_graph_target_{target}.png"),
                reverse=True,
            )
            self._plot_directionality(
                connectivity_metric_matrix[:, target],
                target,
                n_nodes,
                os.path.join(save_path, f"te_graph_target_{target}.png"),
                reverse=True,
                boolean_connectivity=False,
            )

    def plot_connectivity_graph(self, result, save_path=None, show=False):
        """
        Plot connectivity graph
        """
        adj_matrix = result["adjacency_matrix"]
        n_nodes = adj_matrix.shape[0]

        G_1 = nx.DiGraph()

        # position
        for i in range(self.mea.shape[0]):
            for j in range(self.mea.shape[1]):
                center = self.mea[i, j]
                if center < 0:
                    continue
                G_1.add_node(center)

        for source in range(n_nodes):
            if source not in self.mea:
                continue
            for target in range(n_nodes):
                if target not in self.mea:
                    continue
                if source == target:
                    conn = 0
                else:
                    conn = adj_matrix[source, target]
                if np.isnan(conn):
                    conn = 0.0
                G_1.add_edge(source, target, weight=conn)

        return G_1

    def _plot_directionality(
        self,
        connectivity,
        self_index,
        n_nodes,
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
        for i in range(self.mea.shape[0]):
            for j in range(self.mea.shape[1]):
                center = self.mea[i, j]
                if center < 0:
                    continue
                G_1.add_node(center)
                pos[center] = (j, i)

        if boolean_connectivity:
            for idx, conn in enumerate(connectivity):
                if idx == self_index:
                    continue  # No self-connection
                if idx not in self.mea:
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
                if idx not in self.mea:
                    continue  # No connection if node does not exist

                if not reverse:
                    G_1.add_edge(self_index, idx, weight=conn)
                else:
                    G_1.add_edge(idx, self_index, weight=conn)

        # Numbering
        # edge_weights = [G_1[u][v]['weight'] if 'weight' in G_1[u][v] else 0.0 for u, v in G_1.edges()]
        plt.figure()
        widths = nx.get_edge_attributes(G_1, "weight")
        # print(list(widths.items()))
        nx.draw(G_1, pos)
        nx.draw_networkx_edges(
            G_1,
            pos,
            edgelist=widths.keys(),
            width=list(widths.values()),
            alpha=0.5,
            edge_color="lightblue",
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


# @dataclass
# class UndirectedConnectivity(OperatorMixin):
#    pass
