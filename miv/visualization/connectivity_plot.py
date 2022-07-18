import os

import graphviz
import matplotlib.pyplot as plt
import numpy as np
from pyvis.network import Network

from miv.typing import SpikestampsType


def plot_connectivity(
    mea_map: numpy_array,
    connectivity_matrix: numpy_array,
    interactive: bool,
):
    """
    Plots the provided connectivity matrix using the exact location of electrode.

    Parameters
    ----------
    mea_map : numpy array
        array containing spatial location of electrodes
    connectivity_matrix: numpy array
        array containing the connectivity parameters for each pair of electrodes
    interactive: bool
       If set True, the pyvis is used to generate an interactive html plot while False generates a graphviz based plot as pdf

    Returns
    -------
    figure, axes
       matplot figure with bursts plotted for all electrodes
    """

    num_elec = np.size(mea_map)
    elec_mat = np.linspace(1, num_elec, num_elec).astype(int)

    if interactive:
        net = Network(
            height="500px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            notebook="false",
        )

        for i in elec_mat:
            for j in elec_mat:
                if i == j:
                    continue
                else:
                    if connectivity_matrix[int(i - 1), int(j - 1)] != 0 and i < j:
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
                        net.add_node(
                            str(int(j)), x=x2, y=y2, shape="dot", color="#039AFB!"
                        )
                        net.add_edge(str(int(i)), str(int(j)), width="1")

        for n in net.nodes:
            n.update({"physics": False})

        return net.show("nodes.html")

    else:

        g = graphviz.Graph("G", filename="connectivity", engine="neato")
        for i in elec_mat:
            for j in elec_mat:
                if i == j:
                    continue
                else:
                    if connectivity_matrix[int(i - 1), int(j - 1)] != 0 and i < j:

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
        return g.view()
