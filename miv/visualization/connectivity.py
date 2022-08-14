__all__ = ["plot_connectivity", "plot_connectivity_interactive"]

import os

import graphviz
import matplotlib.pyplot as plt
import numpy as np
from pyvis.network import Network

from miv.typing import SpikestampsType


def plot_connectivity(
    mea_map: np.ndarray, connectivity_matrix: np.ndarray, directionality: bool
):
    """
    Plots the provided connectivity matrix with graphviz using the exact location of electrode.

    Parameters
    ----------
    mea_map : np.ndarray
        array containing spatial location of electrodes
    connectivity_matrix: np.ndarray
        array containing the connectivity parameters for each pair of electrodes
    directionality: bool
        if set true can be used to plot directional connectivity plots

    Returns
    -------
    g
        graphviz graph object
    """

    connec_x, connec_y = np.shape(connectivity_matrix)
    assert connec_x == connec_y, "Connectivity matrix should be a square matrix"
    num_elec = np.count_nonzero(mea_map)
    assert num_elec > 0, "Number of electrodes should be greater than 0"
    elec_mat = np.linspace(1, num_elec, num_elec).astype(int)
    assert (
        connec_x == num_elec
    ), "Connectivity matrix should have same dimensions as the number of electrodes"

    if directionality:
        g = graphviz.Digraph("G", filename="connectivity", engine="neato")
    else:
        g = graphviz.Graph("G", filename="connectivity", engine="neato")

    for i in elec_mat:
        for j in elec_mat:
            if i == j:
                continue
            if not directionality and i >= j:
                continue
            if np.isclose(connectivity_matrix[int(i - 1), int(j - 1)], 0):
                continue
            # Register
            ki = int(np.where(mea_map == i)[0][0])
            ji = int(np.where(mea_map == i)[1][0])
            kj = int(np.where(mea_map == j)[0][0])
            jj = int(np.where(mea_map == j)[1][0])

            x1 = str(ki + 0.0)
            y1 = str(ji) + "!"
            posstr1 = x1 + "," + y1

            x2 = str(kj + 0.0)
            y2 = str(jj) + "!"
            posstr2 = x2 + "," + y2
            g.edge(str(i), str(j), color="red")

            g.node(
                str(i),
                pos=posstr1,
                fillcolor="deepskyblue2",
                color="black",
                style="filled,solid",
                shape="circle",
                fontcolor="black",
                fontsize="15",
                fontname="Times:Roman bold",
            )

            g.node(
                str(j),
                pos=posstr2,
                fillcolor="deepskyblue2",
                color="black",
                style="filled,solid",
                shape="circle",
                fontcolor="black",
                fontsize="15",
                fontname="Times:Roman bold",
            )
    return g


def plot_connectivity_interactive(
    mea_map: np.ndarray,
    connectivity_matrix: np.ndarray,
    directionality: bool,
):
    """
    Plots the provided connectivity matrix with pyvis using the exact location of electrode.

    Parameters
    ----------
    mea_map : np.ndarray
        array containing spatial location of electrodes
    connectivity_matrix: np.ndarray
        array containing the connectivity parameters for each pair of electrodes
    directionality: bool
        if set true can be used to plot directional connectivity plots

    Returns
    -------
    net
        pyvis network object
    """
    connec_x, connec_y = np.shape(connectivity_matrix)
    assert connec_x == connec_y, "Connectivity matrix should be a square matrix"
    num_elec = np.count_nonzero(mea_map)
    assert num_elec > 0, "Number of electrodes should be greater than 0"
    elec_mat = np.linspace(1, num_elec, num_elec).astype(int)
    assert (
        connec_x == num_elec
    ), "Connectivity matrix should have same dimensions as the number of electrodes"

    net = Network(
        height="500px",
        width="100%",
        bgcolor="#222222",
        directed=directionality,
        font_color="white",
        notebook="false",
    )
    net.repulsion()

    for i in elec_mat:
        for j in elec_mat:
            if i == j:
                continue
            if np.isclose(connectivity_matrix[int(i - 1), int(j - 1)], 0.0) or i >= j:
                continue
            ki = int(np.where(mea_map == i)[0][0]) * 100
            ji = int(np.where(mea_map == i)[1][0]) * 100
            kj = int(np.where(mea_map == j)[0][0]) * 100
            jj = int(np.where(mea_map == j)[1][0]) * 100

            x1 = str(ki + 0.0)
            y1 = str(ji)

            x2 = str(kj + 0.0)
            y2 = str(jj)

            net.add_node(
                str(int(i)), x=x1, y=y1, shape="dot", color="#039AFB"
            )  # size = size1)
            net.add_node(str(int(j)), x=x2, y=y2, shape="dot", color="#039AFB!")
            net.add_edge(str(int(i)), str(int(j)), width="1")

    for n in net.nodes:
        n.update({"physics": False})

    return net
