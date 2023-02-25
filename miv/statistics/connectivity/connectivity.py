__doc__ = """
Connectivity module
"""
__all__ = ["DirectedConnectivity"]  # , 'UndirectedConnectivity']

from typing import Any, List, Optional, Union

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

    mea: Union[str, np.ndarray]
    channels: Optional[List[int]] = None
    bin_size: float = 0.001
    tag: str = "directional connectivity analysis"
    progress_bar: bool = True

    skip_surrogate: bool = False

    # Surrogate parameters
    surrogate_N: int = 30
    p_threshold: float = 0.05
    seed: int = None

    def __post_init__(self):
        super().__init__()
        if isinstance(self.mea, str):
            self.mea_map = mea_map[self.mea]

    @wrap_generator_to_generator
    def __call__(self, spikestamps: Spikestamps) -> np.ndarray:
        """__call__.

        Parameters
        ----------
        spikestamps : Spikestamps
        """

        binned_spiketrain: Signal = spikestamps.binning(bin_size=self.bin_size)
        n_nodes = (
            binned_spiketrain.number_of_channels
            if self.channels is None
            else len(self.channels)
        )
        channels = self.channels if self.channels is not None else list(range(n_nodes))

        # Get adjacency matrix based on transfer entropy
        adj_matrix = np.zeros([n_nodes, n_nodes], dtype=np.bool_)  # source -> target
        connectivity_metric_matrix = np.zeros(
            [n_nodes, n_nodes], dtype=np.float_
        )  # source -> target
        # TODO: Use mp.Pool
        for sidx, source in tqdm(enumerate(channels), disable=not self.progress_bar):
            if source not in self.mea_map:
                continue
            p_values = []
            metric_values = []
            for tidx, target in tqdm(
                enumerate(channels), disable=not self.progress_bar, leave=False
            ):
                if source != target and target in self.mea_map:
                    source_binned_spiketrain = binned_spiketrain[sidx]
                    target_binned_spiketrain = binned_spiketrain[tidx]
                    p, metrics = self._get_connection_info(
                        source_binned_spiketrain, target_binned_spiketrain
                    )
                else:
                    p, metrics = 1, 0
                p_values.append(p)
                metric_values.append(metrics)
            adj_matrix[sidx] = np.array(p_values) < self.p_threshold
            connectivity_metric_matrix[sidx] = np.array(metric_values)
        connection_ratio = adj_matrix.sum() / adj_matrix.ravel().shape[0]

        info = dict(
            adjacency_matrix=adj_matrix,
            connectivity_matrix=connectivity_metric_matrix,
            connection_ratio=connection_ratio,
        )

        return info

    def _surrogate_t_test(self, source, target, te_history=4, sublength=64, stride=8):
        """
        Surrogate t-test
        """
        assert (
            source.shape[0] == target.shape[0]
        ), f"source.shape={source.shape}, target.shape={target.shape}"
        assert (
            source.shape[0] - sublength > 0
        ), f"source.shape[0]={source.shape[0]}, sublength={sublength}"

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
                pos[center] = (j, i)

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
