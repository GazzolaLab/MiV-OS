__doc__ = """Centrality module"""
__all__ = ["plot_eigenvector_centrality"]

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_eigenvector_centrality(self, result, show=False, save_path=None):
    metric_matrix = result["connectivity_matrix"]
    adjacency_matrix = result["adjacency_matrix"]
    n_nodes = metric_matrix.shape[0]

    # def plot_centrality(self, adjacency_matrix, n_nodes, centrality, mea, ax=None, savename=None, include_colorbar:bool=True):
    # def get_graph(adjacency_matrix, n_nodes):

    # Construct graph
    G = nx.DiGraph()

    # position
    pos = {}
    for i in range(self.mea_map.shape[0]):
        for j in range(self.mea_map.shape[1]):
            center = self.mea_map[i, j]
            if center < 0:
                continue
            G.add_node(center)
            pos[center] = (j, i)

    for source in range(n_nodes):
        if source not in self.mea_map:
            continue
        for target in range(n_nodes):
            if target not in self.mea_map:
                continue
            if source == target:
                conn = 0
            else:
                conn = adjacency_matrix[source, target]
            if np.isnan(conn):
                conn = 0.0
            G.add_edge(source, target, weight=conn)

    # Plotting
    nodes = G.nodes()
    centrality = nx.eigenvector_centrality_numpy(G, weight="weight")
    colors = [centrality[n] for n in nodes]

    # mean = np.nanmean(metric_matrix)
    # std = np.nanstd(metric_matrix)
    # vmin = max(0.0, mean - 2 * std)
    # vmax = mean + 2 * std

    fig = plt.figure()
    plt.imshow(nx.adjacency_matrix(G, weight="weight").todense())
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "connection_matrix.png"))
    plt.close()

    fig = plt.figure()
    ax = plt.gca()

    # nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
    nc = nx.draw_networkx_nodes(
        G, pos, nodelist=nodes, node_color=colors, node_size=100, cmap=plt.cm.jet, ax=ax
    )  # , vmin=0, vmax=0.5)
    for node, (x_coord, y_coord) in pos.items():
        ax.text(x_coord - 0.1, y_coord - 0.1, str(node))
    fig.colorbar(nc, ax=ax)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "eigenvector_centrality.png"))
    plt.close()
